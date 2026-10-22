from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import f1_score

from rareguard.config import load_yaml
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.run_publication_central import (
    Variant,
    build_model,
    classification_objective,
    data_loader,
    effective_class_weights,
    infer,
    load_splits,
    rare_sampling_weights,
    resolve_device,
)
from rareguard.models.publication_models import prototype_compactness_loss
from rareguard.models.torch_models import FocalLoss, supervised_contrastive_loss
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


METHODS = ("RareGuard-FineTune", "RareGuard-ReservoirReplay", "RareGuard-Continual")


def class_incremental_tasks(y: np.ndarray, task_count: int) -> list[list[int]]:
    classes, counts = np.unique(y, return_counts=True)
    ordered = [int(value) for value in classes[np.argsort(-counts)]]
    if len(ordered) < 2:
        raise ValueError("Continual evaluation requires at least two classes.")
    task_count = max(2, min(int(task_count), len(ordered)))
    base_size = max(1, len(ordered) - (task_count - 1))
    tasks = [ordered[:base_size]]
    tasks.extend([[value] for value in ordered[base_size:]])
    if len(tasks) > task_count:
        tail = tasks[task_count - 1 :]
        tasks = tasks[: task_count - 1] + [[value for group in tail for value in group]]
    return tasks


def capped_indices(indices: np.ndarray, y: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= limit:
        return indices.copy()
    selected: list[np.ndarray] = []
    classes = np.unique(y[indices])
    per_class = max(1, limit // len(classes))
    for class_id in classes:
        candidates = indices[y[indices] == class_id]
        selected.append(rng.choice(candidates, size=min(len(candidates), per_class), replace=False))
    result = np.concatenate(selected)
    if len(result) < limit:
        remaining = np.setdiff1d(indices, result, assume_unique=False)
        if len(remaining):
            extra = rng.choice(remaining, size=min(limit - len(result), len(remaining)), replace=False)
            result = np.concatenate([result, extra])
    return rng.permutation(result)


def update_memory(
    method: str,
    seen_indices: np.ndarray,
    y: np.ndarray,
    memory_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "RareGuard-FineTune" or memory_size <= 0:
        return np.asarray([], dtype=int)
    if method == "RareGuard-ReservoirReplay":
        size = min(memory_size, len(seen_indices))
        return rng.choice(seen_indices, size=size, replace=False)
    classes = np.unique(y[seen_indices])
    per_class = max(1, memory_size // len(classes))
    parts = []
    for class_id in classes:
        candidates = seen_indices[y[seen_indices] == class_id]
        parts.append(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False))
    return rng.permutation(np.concatenate(parts))


def seen_validation_f1(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    seen_classes: set[int],
    config: dict,
    device: str,
) -> float:
    mask = np.isin(arrays["y_val"], list(seen_classes))
    if not mask.any():
        return 0.0
    probabilities, _ = infer(
        model,
        arrays["X_val"][mask],
        arrays["y_val"][mask],
        int(config["training"]["batch_size"]),
        device,
        True,
    )
    unseen = sorted(set(range(probabilities.shape[1])) - seen_classes)
    probabilities[:, unseen] = 0.0
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return float(f1_score(arrays["y_val"][mask], probabilities.argmax(axis=1), average="macro", zero_division=0))


def imprint_prototypes(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> None:
    if model.prototype_layer is None or not len(indices):
        return
    model.eval()
    with torch.no_grad():
        _, vectors = infer(model, x[indices], y[indices], batch_size, device, True)
        for class_id in np.unique(y[indices]):
            centroid = torch.tensor(vectors[y[indices] == class_id].mean(axis=0), dtype=torch.float32, device=device)
            model.prototype_layer.weight[int(class_id)].copy_(torch.nn.functional.normalize(centroid, dim=0))


def train_episode(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    seen_classes: set[int],
    config: dict,
    device: str,
) -> tuple[dict[str, torch.Tensor], float]:
    training = config["training"]
    continual = config["continual"]
    classes = int(max(arrays["y_train"].max(), arrays["y_val"].max(), arrays["y_test"].max())) + 1
    weights = effective_class_weights(y[indices], classes, float(training["class_balance_beta"])).to(device)
    criterion = FocalLoss(float(training["focal_gamma"]), weights)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    sampling_weights = rare_sampling_weights(
        y[indices],
        classes,
        float(training.get("rare_sampler_power", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for _ in range(int(continual["epochs_per_task"])):
        model.train()
        for xb, yb in data_loader(
            x[indices],
            y[indices],
            int(training["batch_size"]),
            True,
            sampling_weights,
            0,
        ):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, vectors = model(xb)
            loss = classification_objective(
                logits,
                yb,
                Variant("RareGuard-XIDS", True),
                criterion,
                cross_entropy,
                training,
            )
            loss = loss + float(training["contrastive_weight"]) * supervised_contrastive_loss(
                vectors, yb, float(training["contrastive_temperature"])
            )
            if model.prototype_layer is not None:
                loss = loss + float(training["prototype_weight"]) * prototype_compactness_loss(
                    vectors, yb, model.prototype_layer.weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        score = seen_validation_f1(model, arrays, seen_classes, config, device)
        if score > best_score + 1e-6:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(continual["early_stop_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Continual episode produced no validation checkpoint.")
    model.load_state_dict(best_state)
    return best_state, best_score


def task_validation_scores(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    tasks: list[list[int]],
    completed_task: int,
    seen_classes: set[int],
    config: dict,
    device: str,
) -> list[float]:
    scores = []
    for classes in tasks[: completed_task + 1]:
        mask = np.isin(arrays["y_val"], classes)
        probabilities, _ = infer(
            model,
            arrays["X_val"][mask],
            arrays["y_val"][mask],
            int(config["training"]["batch_size"]),
            device,
            True,
        )
        probabilities[:, sorted(set(range(probabilities.shape[1])) - seen_classes)] = 0.0
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        scores.append(float(f1_score(arrays["y_val"][mask], probabilities.argmax(axis=1), average="macro", zero_division=0)))
    return scores


def run_method(
    dataset: str,
    seed: int,
    method: str,
    arrays: dict[str, np.ndarray],
    config: dict,
    device: str,
    root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    x_train, y_train = arrays["X_train"], arrays["y_train"]
    classes = int(max(y_train.max(), arrays["y_val"].max(), arrays["y_test"].max())) + 1
    tasks = class_incremental_tasks(y_train, int(config["continual"]["tasks"]))
    model = build_model(Variant("RareGuard-XIDS", True), x_train.shape[1], classes, config).to(device)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    seen_classes: set[int] = set()
    seen_indices = np.asarray([], dtype=int)
    memory = np.asarray([], dtype=int)
    trajectories: dict[int, list[float]] = {task_id: [] for task_id in range(len(tasks))}
    stage_rows: list[dict[str, object]] = []
    for task_id, task_classes in enumerate(tasks):
        current = np.flatnonzero(np.isin(y_train, task_classes))
        current = capped_indices(
            current,
            y_train,
            int(config["continual"]["max_current_samples_per_task"]),
            rng,
        )
        train_indices = rng.permutation(np.concatenate([current, memory])) if len(memory) else current
        seen_classes.update(task_classes)
        state, validation_score = train_episode(
            model,
            x_train,
            y_train,
            train_indices,
            arrays,
            seen_classes,
            config,
            device,
        )
        seen_indices = np.unique(np.concatenate([seen_indices, current]))
        memory = update_memory(
            method,
            seen_indices,
            y_train,
            int(config["continual"]["memory_size"]),
            rng,
        )
        if method == "RareGuard-Continual":
            imprint_prototypes(
                model,
                x_train,
                y_train,
                memory,
                int(config["training"]["batch_size"]),
                device,
            )
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        scores = task_validation_scores(model, arrays, tasks, task_id, seen_classes, config, device)
        for observed_task, score in enumerate(scores):
            trajectories[observed_task].append(score)
        stage_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "task": task_id,
                "introduced_classes": json.dumps(task_classes),
                "seen_classes": len(seen_classes),
                "task_train_rows": len(train_indices),
                "memory_rows": len(memory),
                "seen_validation_f1_macro": validation_score,
            }
        )
        peak_rss = max(peak_rss, process.memory_info().rss)
        model.load_state_dict(state)
    test_started = time.perf_counter()
    probabilities, _ = infer(
        model,
        arrays["X_test"],
        arrays["y_test"],
        int(config["training"]["batch_size"]),
        device,
        True,
    )
    test_seconds = time.perf_counter() - test_started
    counts = np.bincount(y_train, minlength=classes)
    cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * len(y_train)))
    rare_ids = {int(index) for index, count in enumerate(counts) if count <= cutoff}
    forgetting = [max(values) - values[-1] for values in trajectories.values() if values]
    model_dir = root / "models" / dataset / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{method}.pt"
    prediction_path = model_dir / f"{method}_locked_test.npz"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, model_path)
    np.savez_compressed(
        prediction_path,
        y=arrays["y_test"],
        probabilities=probabilities,
        prediction=probabilities.argmax(axis=1),
    )
    summary: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "tasks": len(tasks),
        "memory_size": 0 if method == "RareGuard-FineTune" else int(config["continual"]["memory_size"]),
        "mean_validation_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "max_validation_forgetting": float(np.max(forgetting)) if forgetting else 0.0,
        "training_seconds": time.perf_counter() - started,
        "test_seconds": test_seconds,
        "test_samples_per_second": float(len(arrays["y_test"]) / max(test_seconds, 1e-9)),
        "peak_rss_mb": float(peak_rss / (1024 * 1024)),
        "device": device,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "test_access_count": 1,
        **classification_metrics(arrays["y_test"], probabilities, rare_ids),
    }
    return summary, stage_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run class-incremental RareGuard-XIDS memory experiments.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_continual.log")
    device = resolve_device(args.device or config["project"].get("device", "cpu"))
    keys = list(load_yaml(args.datasets)["datasets"]) if args.dataset == "all" else [args.dataset]
    summary_path = root / "results" / "continual_seed_metrics.csv"
    stages_path = root / "results" / "continual_task_metrics.csv"
    summaries = pd.read_csv(summary_path).to_dict("records") if args.resume and summary_path.exists() else []
    stages = pd.read_csv(stages_path).to_dict("records") if args.resume and stages_path.exists() else []
    complete = {(str(row["dataset"]), int(row["seed"]), str(row["method"])) for row in summaries}
    for dataset in keys:
        arrays = load_splits(root / "processed" / dataset / "splits.npz")
        metadata = json.loads((root / "processed" / dataset / "split_metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("ready"):
            raise RuntimeError(f"Publication data gate failed for {dataset}")
        for seed in [int(value) for value in config["continual"]["seeds"]]:
            for method in METHODS:
                if (dataset, seed, method) in complete:
                    logger.info("Skipping completed %s seed=%d %s", dataset, seed, method)
                    continue
                logger.info("Starting %s seed=%d %s device=%s", dataset, seed, method, device)
                summary, task_rows = run_method(dataset, seed, method, arrays, config, device, root)
                summaries.append(summary)
                stages.extend(task_rows)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(summaries).to_csv(summary_path, index=False)
                pd.DataFrame(stages).to_csv(stages_path, index=False)
                logger.info(
                    "%s seed=%d %s test_f1_macro=%.4f rare_f1=%.4f forgetting=%.4f",
                    dataset,
                    seed,
                    method,
                    summary["f1_macro"],
                    summary["rare_f1_macro"],
                    summary["mean_validation_forgetting"],
                )
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
