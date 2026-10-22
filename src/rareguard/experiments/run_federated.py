from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from rareguard.config import load_config, load_yaml
from rareguard.data.splits import load_npz, stratified_split
from rareguard.federated.partition import dirichlet_non_iid_partition
from rareguard.models.torch_models import FalconIDNet, FocalLoss, RareGuardNet, supervised_contrastive_loss
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


@dataclass(frozen=True)
class ClientUpdate:
    state: dict[str, torch.Tensor]
    samples: int
    rare_share: float
    energy_budget: float


def make_model(model_name: str, input_dim: int, classes: int, latent_dim: int, hidden_dim: int, dropout: float) -> torch.nn.Module:
    if model_name == "RareGuard":
        return RareGuardNet(input_dim, classes, latent_dim, hidden_dim, dropout)
    return FalconIDNet(input_dim, classes, latent_dim, hidden_dim, dropout)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_local(
    model_name: str,
    global_state: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    classes: int,
    cfg,
    device: str,
    fedprox_mu: float = 0.0,
) -> dict[str, torch.Tensor]:
    model = make_model(
        model_name,
        x.shape[1],
        classes,
        int(cfg.raw["embedding"]["latent_dim"]),
        int(cfg.raw["embedding"]["hidden_dim"]),
        float(cfg.raw["embedding"]["dropout"]),
    ).to(device)
    model.load_state_dict(global_state)
    global_params = {k: v.detach().clone().to(device) for k, v in global_state.items()}
    criterion = FocalLoss(float(cfg.raw["embedding"]["focal_gamma"])) if model_name == "RareGuard" else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.raw["training"]["learning_rate"]), weight_decay=float(cfg.raw["training"]["weight_decay"]))
    for _ in range(int(cfg.raw["federated"]["local_epochs"])):
        model.train()
        for xb, yb in loader(x, y, int(cfg.raw["training"]["batch_size"]), True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if model_name == "RareGuard":
                logits, z = model(xb)
                loss = criterion(logits, yb) + 0.1 * supervised_contrastive_loss(z, yb, float(cfg.raw["embedding"]["contrastive_temperature"]))
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            if fedprox_mu:
                prox = torch.zeros((), device=device)
                for name, param in model.named_parameters():
                    prox = prox + torch.sum((param - global_params[name]) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox
            loss.backward()
            optimizer.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def average_states(updates: list[ClientUpdate], weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No client updates to aggregate.")
    if weights is None:
        weights = [float(update.samples) for update in updates]
    total = float(sum(weights))
    averaged: dict[str, torch.Tensor] = {}
    for key in updates[0].state:
        if not torch.is_floating_point(updates[0].state[key]):
            averaged[key] = updates[0].state[key].clone()
            continue
        value = torch.zeros_like(updates[0].state[key], dtype=torch.float32)
        for update, weight in zip(updates, weights):
            value += update.state[key].float() * (float(weight) / total)
        averaged[key] = value
    return averaged


def clustered_average(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    if len(updates) < 2:
        return average_states(updates)
    median_rare = float(np.median([u.rare_share for u in updates]))
    low = [u for u in updates if u.rare_share <= median_rare]
    high = [u for u in updates if u.rare_share > median_rare]
    groups = [group for group in [low, high] if group]
    group_updates = [
        ClientUpdate(
            average_states(group),
            sum(u.samples for u in group),
            float(np.mean([u.rare_share for u in group])),
            float(np.mean([u.energy_budget for u in group])),
        )
        for group in groups
    ]
    return average_states(group_updates)


def evaluate_state(
    state: dict[str, torch.Tensor],
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    classes: int,
    cfg,
    device: str,
) -> dict[str, float]:
    model = make_model(
        model_name,
        x.shape[1],
        classes,
        int(cfg.raw["embedding"]["latent_dim"]),
        int(cfg.raw["embedding"]["hidden_dim"]),
        float(cfg.raw["embedding"]["dropout"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader(x, y, 2048, False):
            xb = xb.to(device)
            if model_name == "RareGuard":
                logits, _ = model(xb)
            else:
                logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def run_method(
    dataset: str,
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg,
    seed: int,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    set_seed(seed)
    clients = int(cfg.raw["federated"]["clients"])
    rounds = int(cfg.raw["federated"]["rounds"])
    alpha = float(cfg.raw["federated"]["non_iid_alpha"])
    dropout_rate = float(cfg.raw["federated"]["dropout_rate"])
    rng = np.random.default_rng(seed)
    classes = int(max(np.max(y_train), np.max(y_test))) + 1
    partitions = dirichlet_non_iid_partition(y_train, clients, alpha, seed)
    counts = np.bincount(y_train)
    rare_cutoff = max(1, int(float(cfg.raw["data"]["min_rare_class_ratio"]) * len(y_train)))
    rare_ids = {int(i) for i, c in enumerate(counts) if c <= rare_cutoff}
    model_name = "RareGuard" if method == "RareGuardFed" else "FalconID"
    global_model = make_model(
        model_name,
        x_train.shape[1],
        classes,
        int(cfg.raw["embedding"]["latent_dim"]),
        int(cfg.raw["embedding"]["hidden_dim"]),
        float(cfg.raw["embedding"]["dropout"]),
    ).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    round_rows: list[dict[str, object]] = []
    participating_total = 0

    for round_id in range(1, rounds + 1):
        updates: list[ClientUpdate] = []
        for idx in partitions:
            if len(idx) == 0 or rng.random() < dropout_rate:
                continue
            client_y = y_train[idx]
            rare_share = float(np.isin(client_y, list(rare_ids)).mean()) if rare_ids else 0.0
            energy_budget = float(rng.uniform(0.45, 1.0))
            local_state = train_local(
                model_name,
                global_state,
                x_train[idx],
                client_y,
                classes,
                cfg,
                device,
                fedprox_mu=0.01 if method == "FedProx" else 0.0,
            )
            updates.append(ClientUpdate(local_state, len(idx), rare_share, energy_budget))
        if not updates:
            continue
        participating_total += len(updates)
        if method == "CFL":
            global_state = clustered_average(updates)
        elif method == "RareGuardFed":
            weights = [u.samples * (0.5 + u.energy_budget) * (1.0 + u.rare_share) for u in updates]
            global_state = average_states(updates, weights)
        else:
            global_state = average_states(updates)
        metrics = evaluate_state(global_state, model_name, x_test, y_test, classes, cfg, device)
        round_rows.append({"dataset": dataset, "method": method, "round": round_id, "participating_clients": len(updates), **metrics})

    final_metrics: dict[str, object]
    final_metrics = round_rows[-1] if round_rows else {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}
    summary = {
        "dataset": dataset,
        "method": method,
        "clients": clients,
        "rounds": rounds,
        "non_iid_alpha": alpha,
        "dropout_rate": dropout_rate,
        "avg_participating_clients": float(participating_total / max(1, rounds)),
        "communication_units": int(participating_total),
        "model_family": model_name,
        "evidence": "measured_federated_training",
        "accuracy": final_metrics["accuracy"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "f1_macro": final_metrics["f1_macro"],
        "f1_weighted": final_metrics["f1_weighted"],
    }
    return summary, round_rows


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_checkpoints(
    summaries: list[dict[str, object]],
    rounds: list[dict[str, object]],
    summary_path: Path,
    rounds_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(rounds).to_csv(rounds_path, index=False)


def completed_pairs(
    summaries: list[dict[str, object]],
    round_rows: list[dict[str, object]],
    expected_rounds: int,
) -> set[tuple[str, str]]:
    round_counts: dict[tuple[str, str], int] = {}
    for row in round_rows:
        pair = (str(row["dataset"]), str(row["method"]))
        round_counts[pair] = round_counts.get(pair, 0) + 1
    return {
        (str(row["dataset"]), str(row["method"]))
        for row in summaries
        if round_counts.get((str(row["dataset"]), str(row["method"])), 0) >= expected_rounds
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CPU-first federated IDS experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue from completed dataset-method checkpoints.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = "cuda" if (args.device or cfg.device) == "cuda" and torch.cuda.is_available() else "cpu"
    logger = configure_logging(cfg.artifact_root / "logs" / "run_federated.log")
    keys = list(load_yaml(args.datasets)["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    out = cfg.artifact_root / "results" / "federated_summary.csv"
    round_out = cfg.artifact_root / "results" / "federated_round_metrics.csv"
    summaries = read_rows(out) if args.resume else []
    rounds = read_rows(round_out) if args.resume else []
    expected_rounds = int(cfg.raw["federated"]["rounds"])
    completed = completed_pairs(summaries, rounds, expected_rounds)
    logger.info(
        "Federated run device=%s resume=%s completed_pairs=%d",
        device,
        args.resume,
        len(completed),
    )
    for key in keys:
        x, y = load_npz(cfg.artifact_root / "processed" / key / "features.npz")
        split = stratified_split(x, y, cfg.raw["data"]["test_size"], cfg.raw["data"]["validation_size"], cfg.seed)
        for method in cfg.raw["federated"]["methods"]:
            pair = (key, method)
            if pair in completed:
                logger.info("Skipping completed %s %s", key, method)
                continue
            summaries = [row for row in summaries if (str(row["dataset"]), str(row["method"])) != pair]
            rounds = [row for row in rounds if (str(row["dataset"]), str(row["method"])) != pair]
            logger.info("Starting %s %s", key, method)
            try:
                summary, round_rows = run_method(
                    key,
                    method,
                    split.x_train,
                    split.y_train,
                    split.x_test,
                    split.y_test,
                    cfg,
                    cfg.seed,
                    device,
                )
            except Exception:
                logger.exception("Failed %s %s", key, method)
                write_checkpoints(summaries, rounds, out, round_out)
                raise
            summaries.append(summary)
            rounds.extend(round_rows)
            write_checkpoints(summaries, rounds, out, round_out)
            logger.info("%s %s f1_macro=%.4f", key, method, summary["f1_macro"])
            del summary, round_rows
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_checkpoints(summaries, rounds, out, round_out)
    logger.info("Wrote %s and %s", out, round_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
