from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

from rareguard.config import load_yaml
from rareguard.experiments.adaptive_config import apply_dataset_hyperparameters
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.run_publication_central import (
    Variant,
    apply_temperature,
    build_model,
    classification_objective,
    data_loader,
    effective_class_weights,
    fit_temperature,
    infer,
    load_splits,
    rare_sampling_weights,
    resolve_device,
)
from rareguard.federated.partition import dirichlet_non_iid_partition
from rareguard.models.publication_models import prototype_compactness_loss
from rareguard.models.torch_models import FocalLoss, supervised_contrastive_loss
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


@dataclass(frozen=True)
class Update:
    state: dict[str, torch.Tensor]
    samples: int
    rare_share: float
    samples_per_second: float
    delta: np.ndarray


METHODS = ("RareGuard-FedAvg", "RareGuard-FedProx", "RareGuard-CFL", "RareGuardFed")


def average(updates: list[Update], weights: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    if weights is None:
        weights = np.asarray([item.samples for item in updates], dtype=float)
    weights = weights / weights.sum()
    result: dict[str, torch.Tensor] = {}
    for name in updates[0].state:
        first = updates[0].state[name]
        if not torch.is_floating_point(first):
            result[name] = first.clone()
            continue
        result[name] = sum(item.state[name].float() * float(weight) for item, weight in zip(updates, weights))
    return result


def flatten_delta(local: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [(local[name].float() - global_state[name].float()).reshape(-1).numpy() for name in local if torch.is_floating_point(local[name])]
    )


def train_client(
    global_state: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    classes: int,
    config: dict,
    device: str,
    proximal_mu: float,
) -> tuple[dict[str, torch.Tensor], float]:
    variant = Variant("RareGuard-XIDS", True)
    model = build_model(variant, x.shape[1], classes, config).to(device)
    model.load_state_dict(global_state)
    original = {name: value.to(device) for name, value in global_state.items()}
    training = config["training"]
    weights = effective_class_weights(y, classes, float(training["class_balance_beta"])).to(device)
    criterion = FocalLoss(float(training["focal_gamma"]), weights)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    sampling_weights = rare_sampling_weights(y, classes, float(training.get("rare_sampler_power", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    started = time.perf_counter()
    for _ in range(int(config["federated"]["local_epochs"])):
        model.train()
        for xb, yb in data_loader(x, y, int(training["batch_size"]), True, sampling_weights, 0):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, z = model(xb)
            loss = classification_objective(logits, yb, variant, criterion, cross_entropy, training)
            loss = loss + float(training["contrastive_weight"]) * supervised_contrastive_loss(
                z, yb, float(training["contrastive_temperature"])
            )
            if model.prototype_layer is not None:
                loss = loss + float(training["prototype_weight"]) * prototype_compactness_loss(z, yb, model.prototype_layer.weight)
            if proximal_mu > 0:
                proximity = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    proximity = proximity + torch.sum((parameter - original[name]) ** 2)
                loss = loss + 0.5 * proximal_mu * proximity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    elapsed = time.perf_counter() - started
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, elapsed


def validation_f1(state: dict[str, torch.Tensor], arrays: dict[str, np.ndarray], config: dict, device: str) -> float:
    classes = int(max(arrays["y_train"].max(), arrays["y_val"].max(), arrays["y_test"].max())) + 1
    variant = Variant("RareGuard-XIDS", True)
    model = build_model(variant, arrays["X_train"].shape[1], classes, config).to(device)
    model.load_state_dict(state)
    probabilities, _ = infer(model, arrays["X_val"], arrays["y_val"], int(config["training"]["batch_size"]), device, True)
    return float(f1_score(arrays["y_val"], probabilities.argmax(axis=1), average="macro", zero_division=0))


def aggregate_method(method: str, updates: list[Update], arrays: dict[str, np.ndarray], config: dict, device: str) -> dict[str, torch.Tensor]:
    if method == "RareGuard-CFL" and len(updates) >= 3:
        deltas = np.stack([item.delta for item in updates])
        labels = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(deltas)
        candidates = [average([item for item, label in zip(updates, labels) if label == cluster]) for cluster in np.unique(labels)]
        return max(candidates, key=lambda state: validation_f1(state, arrays, config, device))
    if method == "RareGuardFed":
        throughput = np.asarray([item.samples_per_second for item in updates], dtype=float)
        throughput = throughput / max(float(np.median(throughput)), 1e-9)
        throughput = np.clip(throughput, 0.5, 2.0)
        weights = np.asarray(
            [item.samples * (1.0 + item.rare_share) * math.sqrt(float(speed)) for item, speed in zip(updates, throughput)],
            dtype=float,
        )
        return average(updates, weights)
    return average(updates)


def run_method(dataset: str, seed: int, method: str, arrays: dict[str, np.ndarray], config: dict, device: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    set_seed(seed)
    x_train, y_train = arrays["X_train"], arrays["y_train"]
    classes = int(max(y_train.max(), arrays["y_val"].max(), arrays["y_test"].max())) + 1
    model = build_model(Variant("RareGuard-XIDS", True), x_train.shape[1], classes, config).to(device)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    partitions = dirichlet_non_iid_partition(
        y_train,
        int(config["federated"]["clients"]),
        float(config["federated"]["non_iid_alpha"]),
        seed,
    )
    counts = np.bincount(y_train, minlength=classes)
    cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * len(y_train)))
    rare_ids = {int(i) for i, count in enumerate(counts) if count <= cutoff}
    rng = np.random.default_rng(seed)
    best_state = state
    best_validation = -1.0
    rows: list[dict[str, object]] = []
    communication_bytes = 0
    model_bytes = sum(value.numel() * value.element_size() for value in state.values())
    started = time.perf_counter()
    for round_id in range(1, int(config["federated"]["rounds"]) + 1):
        updates: list[Update] = []
        for indices in partitions:
            if len(indices) == 0 or rng.random() < float(config["federated"]["dropout_rate"]):
                continue
            local_state, elapsed = train_client(
                state,
                x_train[indices],
                y_train[indices],
                classes,
                config,
                device,
                float(config["federated"]["proximal_mu"]) if method == "RareGuard-FedProx" else 0.0,
            )
            updates.append(
                Update(
                    local_state,
                    len(indices),
                    float(np.isin(y_train[indices], list(rare_ids)).mean()) if rare_ids else 0.0,
                    float(len(indices) / max(elapsed, 1e-9)),
                    flatten_delta(local_state, state),
                )
            )
        if not updates:
            continue
        communication_bytes += 2 * len(updates) * model_bytes
        state = aggregate_method(method, updates, arrays, config, device)
        score = validation_f1(state, arrays, config, device)
        rows.append({"dataset": dataset, "seed": seed, "method": method, "round": round_id, "participants": len(updates), "validation_f1_macro": score, "communication_bytes_cumulative": communication_bytes})
        if score > best_validation:
            best_validation = score
            best_state = {name: value.clone() for name, value in state.items()}
    model.load_state_dict(best_state)
    validation_probabilities, _ = infer(model, arrays["X_val"], arrays["y_val"], int(config["training"]["batch_size"]), device, True)
    temperature = fit_temperature(validation_probabilities, arrays["y_val"])
    raw_test_probabilities, _ = infer(model, arrays["X_test"], arrays["y_test"], int(config["training"]["batch_size"]), device, True)
    test_probabilities = apply_temperature(raw_test_probabilities, temperature)
    raw_ece = classification_metrics(arrays["y_test"], raw_test_probabilities, rare_ids)["ece_15"]
    summary: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "backbone": "RareGuard-XIDS",
        "best_validation_f1_macro": best_validation,
        "rounds": int(config["federated"]["rounds"]),
        "clients": int(config["federated"]["clients"]),
        "non_iid_alpha": float(config["federated"]["non_iid_alpha"]),
        "communication_bytes": communication_bytes,
        "wall_seconds": time.perf_counter() - started,
        "test_access_count": 1,
        "device": device,
        "temperature": temperature,
        "raw_ece_15": raw_ece,
        **classification_metrics(arrays["y_test"], test_probabilities, rare_ids),
    }
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fair same-backbone federated publication experiments.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_federated.log")
    device = resolve_device(args.device or config["project"].get("device", "cpu"))
    keys = list(load_yaml(args.datasets)["datasets"]) if args.dataset == "all" else [args.dataset]
    summary_path = root / "results" / "federated_seed_metrics.csv"
    rounds_path = root / "results" / "federated_validation_rounds.csv"
    summaries = pd.read_csv(summary_path).to_dict("records") if args.resume and summary_path.exists() else []
    rounds = pd.read_csv(rounds_path).to_dict("records") if args.resume and rounds_path.exists() else []
    complete = {(str(row["dataset"]), int(row["seed"]), str(row["method"])) for row in summaries}
    for dataset in keys:
        dataset_config = apply_dataset_hyperparameters(config, root, dataset)
        arrays = load_splits(root / "processed" / dataset / "splits.npz")
        metadata = json.loads((root / "processed" / dataset / "split_metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("ready"):
            raise RuntimeError(f"Publication data gate failed for {dataset}")
        for seed in [int(value) for value in config["federated"].get("seeds", config["project"]["seeds"])]:
            for method in METHODS:
                if (dataset, seed, method) in complete:
                    continue
                logger.info("Starting %s seed=%d %s", dataset, seed, method)
                summary, run_rows = run_method(dataset, seed, method, arrays, dataset_config, device)
                summaries.append(summary)
                rounds.extend(run_rows)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(summaries).to_csv(summary_path, index=False)
                pd.DataFrame(rounds).to_csv(rounds_path, index=False)
                logger.info("%s seed=%d %s f1_macro=%.4f", dataset, seed, method, summary["f1_macro"])
                gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
