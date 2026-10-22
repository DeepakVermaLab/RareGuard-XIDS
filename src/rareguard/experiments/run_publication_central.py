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
import psutil
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rareguard.config import load_yaml
from rareguard.evaluation.metrics import classification_metrics
from rareguard.models.publication_models import NovaXIDSNet, RareGuardXIDSNet, prototype_compactness_loss
from rareguard.models.torch_models import FocalLoss, supervised_contrastive_loss
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


@dataclass(frozen=True)
class Variant:
    label: str
    rareguard: bool
    focal: bool = True
    contrastive: bool = True
    prototypes: bool = True
    class_balance: bool = True


def resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_splits(path: Path) -> dict[str, np.ndarray]:
    values = np.load(path, allow_pickle=True)
    return {name: values[name] for name in values.files}


def data_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sample_weights: np.ndarray | None = None,
    seed: int | None = None,
) -> DataLoader:
    sampler = None
    if sample_weights is not None:
        generator = torch.Generator().manual_seed(int(seed or 0))
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
    )


def rare_sampling_weights(y: np.ndarray, classes: int, power: float) -> np.ndarray | None:
    if power <= 0:
        return None
    counts = np.bincount(y, minlength=classes).astype(float)
    weights = np.power(np.maximum(counts[y], 1.0), -power)
    return weights / weights.mean()


def classification_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    variant: Variant,
    focal: FocalLoss,
    cross_entropy: torch.nn.CrossEntropyLoss,
    training: dict,
) -> torch.Tensor:
    if not variant.focal:
        return cross_entropy(logits, targets)
    if not variant.rareguard:
        return focal(logits, targets)
    focal_mix = float(training.get("focal_mix", 1.0))
    return focal_mix * focal(logits, targets) + (1.0 - focal_mix) * cross_entropy(logits, targets)


def effective_class_weights(y: np.ndarray, classes: int, beta: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(float)
    effective = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def infer(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: str, rareguard: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in data_loader(x, y, batch_size, False):
            xb = xb.to(device)
            if rareguard:
                logits, z = model(xb)
            else:
                logits = model(xb)
                z = model.encode(xb)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            vectors.append(z.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(vectors)


def build_model(variant: Variant, input_dim: int, classes: int, config: dict) -> torch.nn.Module:
    common = {
        "input_dim": input_dim,
        "num_classes": classes,
        "latent_dim": int(config["model"]["latent_dim"]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "dropout": float(config["model"]["dropout"]),
    }
    if variant.rareguard:
        return RareGuardXIDSNet(**common, use_prototypes=variant.prototypes)
    return NovaXIDSNet(**common)


def train_one(
    dataset: str,
    seed: int,
    variant: Variant,
    arrays: dict[str, np.ndarray],
    config: dict,
    output_root: Path,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    set_seed(seed)
    x_train, y_train = arrays["X_train"], arrays["y_train"]
    x_val, y_val = arrays["X_val"], arrays["y_val"]
    x_test, y_test = arrays["X_test"], arrays["y_test"]
    classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    model = build_model(variant, x_train.shape[1], classes, config).to(device)
    training = config["training"]
    weights = effective_class_weights(y_train, classes, float(training["class_balance_beta"])).to(device)
    objective_weights = weights if variant.class_balance else None
    focal = FocalLoss(float(training["focal_gamma"]), objective_weights)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=objective_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    sampling_weights = rare_sampling_weights(
        y_train,
        classes,
        float(training.get("rare_sampler_power", 0.0)) if variant.rareguard else 0.0,
    )
    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in data_loader(x_train, y_train, batch_size, True, sampling_weights, seed + epoch):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if variant.rareguard:
                logits, z = model(xb)
                loss = classification_objective(logits, yb, variant, focal, cross_entropy, training)
                if variant.contrastive:
                    loss = loss + float(training["contrastive_weight"]) * supervised_contrastive_loss(
                        z, yb, float(training["contrastive_temperature"])
                    )
                if variant.prototypes and model.prototype_layer is not None:
                    loss = loss + float(training["prototype_weight"]) * prototype_compactness_loss(
                        z, yb, model.prototype_layer.weight
                    )
            else:
                logits = model(xb)
                loss = classification_objective(logits, yb, variant, focal, cross_entropy, training)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probabilities, _ = infer(model, x_val, y_val, batch_size, device, variant.rareguard)
        val_f1 = float(f1_score(y_val, val_probabilities.argmax(axis=1), average="macro", zero_division=0))
        history.append({"dataset": dataset, "seed": seed, "model": variant.label, "epoch": epoch, "loss": float(np.mean(losses)), "val_f1_macro": val_f1})
        peak_rss = max(peak_rss, process.memory_info().rss)
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(training["early_stop_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    test_started = time.perf_counter()
    probabilities, vectors = infer(model, x_test, y_test, batch_size, device, variant.rareguard)
    test_seconds = time.perf_counter() - test_started
    counts = np.bincount(y_train, minlength=classes)
    rare_cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * len(y_train)))
    rare_ids = {int(i) for i, count in enumerate(counts) if count <= rare_cutoff}
    metrics = classification_metrics(y_test, probabilities, rare_ids)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=np.arange(classes), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "dataset": dataset,
            "seed": seed,
            "model": variant.label,
            "class_id": np.arange(classes),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "train_count": counts,
            "is_rare": [i in rare_ids for i in range(classes)],
        }
    )
    model_dir = output_root / "models" / dataset / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{variant.label}.pt"
    vector_path = model_dir / f"{variant.label}_locked_test.npz"
    torch.save(best_state, model_path)
    np.savez_compressed(vector_path, y=y_test, probabilities=probabilities, prediction=predictions, vectors=vectors)
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "model": variant.label,
        "best_epoch": best_epoch,
        "best_validation_f1_macro": best_f1,
        "training_seconds": time.perf_counter() - started,
        "test_seconds": test_seconds,
        "test_samples_per_second": float(len(y_test) / max(test_seconds, 1e-9)),
        "peak_rss_mb": float(peak_rss / (1024 * 1024)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": device,
        "classification_objective": (
            f"hybrid_focal_ce_{float(training.get('focal_mix', 1.0)):.3f}"
            if variant.rareguard and variant.focal
            else ("class_balanced_focal" if variant.focal else "class_balanced_cross_entropy")
        ),
        "rare_sampler_power": float(training.get("rare_sampler_power", 0.0)) if variant.rareguard else 0.0,
        "contrastive_objective": bool(variant.rareguard and variant.contrastive),
        "prototype_objective": bool(variant.rareguard and variant.prototypes),
        "class_balanced_objective": bool(variant.class_balance),
        "model_path": str(model_path),
        "prediction_path": str(vector_path),
        "test_access_count": 1,
        **metrics,
    }
    return row, history, per_class


def variants(config: dict) -> list[Variant]:
    result = [Variant("NOVA-XIDS", rareguard=False), Variant("RareGuard-XIDS", rareguard=True)]
    requested = set(config["protocol"].get("ablations", []))
    if "no_class_balance" in requested:
        result.append(Variant("RareGuard-no-class-balance", True, class_balance=False))
    if "no_contrastive" in requested:
        result.append(Variant("RareGuard-no-contrastive", True, contrastive=False))
    if "no_prototype" in requested:
        result.append(Variant("RareGuard-no-prototype", True, prototypes=False))
    return result


def aggregate_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    identifier = {"dataset", "seed", "model", "model_path", "prediction_path", "device"}
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in identifier]
    rows: list[dict[str, object]] = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        row: dict[str, object] = {"dataset": dataset, "model": model, "seeds": int(group["seed"].nunique())}
        for column in numeric:
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            mean = float(values.mean())
            half = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
            row[f"{column}_mean"] = mean
            row[f"{column}_ci95_half"] = half
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run repeated locked-test NOVA-XIDS and RareGuard-XIDS experiments.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    output_root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(output_root / "logs" / "run_publication_central.log")
    device = resolve_device(args.device or config["project"].get("device", "cpu"))
    dataset_config = load_yaml(args.datasets)["datasets"]
    keys = list(dataset_config) if args.dataset == "all" else [args.dataset]
    result_path = output_root / "results" / "central_seed_metrics.csv"
    history_path = output_root / "results" / "central_training_history.csv"
    class_path = output_root / "results" / "central_per_class_metrics.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(result_path) if args.resume and result_path.exists() else pd.DataFrame()
    results = existing.to_dict("records")
    histories = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    class_rows = pd.read_csv(class_path).to_dict("records") if args.resume and class_path.exists() else []
    complete = {(str(row["dataset"]), int(row["seed"]), str(row["model"])) for row in results}
    for dataset in keys:
        split_path = output_root / "processed" / dataset / "splits.npz"
        metadata_path = split_path.with_name("split_metadata.json")
        if not split_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Publication split missing for {dataset}; run prepare_publication_data first.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("ready"):
            raise RuntimeError(f"Leakage/readiness gate failed for {dataset}: {metadata.get('duplicate_hash_overlap')}")
        arrays = load_splits(split_path)
        for seed in [int(value) for value in config["project"]["seeds"]]:
            for variant in variants(config):
                if variant.label not in {"NOVA-XIDS", "RareGuard-XIDS"} and seed not in {
                    int(value) for value in config["protocol"].get("ablation_seeds", [])
                }:
                    continue
                key = (dataset, seed, variant.label)
                if key in complete:
                    logger.info("Skipping completed %s seed=%d %s", dataset, seed, variant.label)
                    continue
                logger.info("Starting %s seed=%d %s device=%s", dataset, seed, variant.label, device)
                row, run_history, per_class = train_one(dataset, seed, variant, arrays, config, output_root, device)
                results.append(row)
                histories.extend(run_history)
                class_rows.extend(per_class.to_dict("records"))
                pd.DataFrame(results).to_csv(result_path, index=False)
                pd.DataFrame(histories).to_csv(history_path, index=False)
                pd.DataFrame(class_rows).to_csv(class_path, index=False)
                aggregate_seed_metrics(pd.DataFrame(results)).to_csv(output_root / "results" / "central_summary_ci.csv", index=False)
                logger.info("%s seed=%d %s test_f1_macro=%.4f rare_f1=%.4f", dataset, seed, variant.label, row["f1_macro"], row["rare_f1_macro"])
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
