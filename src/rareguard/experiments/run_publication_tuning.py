from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def train_validation_candidate(
    arrays: dict[str, np.ndarray],
    config: dict,
    seed: int,
    device: str,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    set_seed(seed)
    x_train, y_train = arrays["X_train"], arrays["y_train"]
    x_val, y_val = arrays["X_val"], arrays["y_val"]
    classes = int(max(y_train.max(), y_val.max())) + 1
    variant = Variant("RareGuard-XIDS", True)
    model = build_model(variant, x_train.shape[1], classes, config).to(device)
    training = config["training"]
    weights = effective_class_weights(y_train, classes, float(training["class_balance_beta"])).to(device)
    focal = FocalLoss(float(training["focal_gamma"]), weights)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sample_weights = rare_sampling_weights(y_train, classes, float(training.get("rare_sampler_power", 0.0)))
    counts = np.bincount(y_train, minlength=classes)
    rare_cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * len(y_train)))
    rare_ids = {int(index) for index, count in enumerate(counts) if count <= rare_cutoff}
    best_score = -1.0
    best_row: dict[str, float | int] | None = None
    history: list[dict[str, float | int]] = []
    patience = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        losses = []
        for xb, yb in data_loader(
            x_train,
            y_train,
            int(training["batch_size"]),
            True,
            sample_weights,
            seed + epoch,
        ):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, vectors = model(xb)
            loss = classification_objective(logits, yb, variant, focal, cross_entropy, training)
            loss = loss + float(training["contrastive_weight"]) * supervised_contrastive_loss(
                vectors,
                yb,
                float(training["contrastive_temperature"]),
            )
            if model.prototype_layer is not None:
                loss = loss + float(training["prototype_weight"]) * prototype_compactness_loss(
                    vectors,
                    yb,
                    model.prototype_layer.weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probabilities, _ = infer(model, x_val, y_val, int(training["batch_size"]), device, True)
        metrics = classification_metrics(y_val, probabilities, rare_ids)
        rare_f1 = metrics["rare_f1_macro"]
        selection_score = float(metrics["f1_macro"] if np.isnan(rare_f1) else 0.5 * (metrics["f1_macro"] + rare_f1))
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_selection_score": selection_score,
            "validation_f1_macro": metrics["f1_macro"],
            "validation_rare_f1_macro": rare_f1,
        }
        history.append(row)
        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_row = row.copy()
            patience = 0
        else:
            patience += 1
            if patience >= int(training["early_stop_patience"]):
                break
    if best_row is None:
        raise RuntimeError("Validation tuning produced no checkpoint.")
    return best_row, history


def candidate_grid(config: dict) -> list[dict[str, float]]:
    tuning = config["tuning"]
    return [
        {
            "focal_mix": float(focal_mix),
            "rare_sampler_power": float(sampler_power),
            "contrastive_weight": float(contrastive_weight),
        }
        for focal_mix, sampler_power, contrastive_weight in itertools.product(
            tuning["focal_mix"],
            tuning["rare_sampler_power"],
            tuning["contrastive_weight"],
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-only RareGuard hyperparameter selection.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--dataset", default="cse_cic_ids2018")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    development = set(config["protocol"].get("development_datasets", []))
    if args.dataset not in development:
        raise ValueError(f"Tuning is restricted to declared development datasets: {sorted(development)}")
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_tuning.log")
    device = resolve_device(args.device or config["project"].get("device", "cpu"))
    arrays = load_splits(root / "processed" / args.dataset / "splits.npz")
    result_path = root / "results" / "validation_tuning_candidates.csv"
    history_path = root / "results" / "validation_tuning_history.csv"
    rows = pd.read_csv(result_path).to_dict("records") if args.resume and result_path.exists() else []
    histories = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    complete = {
        (
            float(row["focal_mix"]),
            float(row["rare_sampler_power"]),
            float(row["contrastive_weight"]),
            int(row["seed"]),
        )
        for row in rows
    }
    seeds = [int(value) for value in config["tuning"]["seeds"]]
    candidates = candidate_grid(config)
    for candidate_id, candidate in enumerate(candidates):
        for seed in seeds:
            candidate_key = (
                candidate["focal_mix"],
                candidate["rare_sampler_power"],
                candidate["contrastive_weight"],
                seed,
            )
            if candidate_key in complete:
                continue
            candidate_config = copy.deepcopy(config)
            candidate_config["training"].update(candidate)
            logger.info("candidate=%d seed=%d settings=%s", candidate_id, seed, candidate)
            best, history = train_validation_candidate(arrays, candidate_config, seed, device)
            row = {
                "dataset": args.dataset,
                "candidate_id": candidate_id,
                "seed": seed,
                "device": device,
                **candidate,
                **best,
                "test_access_count": 0,
            }
            rows.append(row)
            histories.extend({"dataset": args.dataset, "candidate_id": candidate_id, "seed": seed, **candidate, **item} for item in history)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(result_path, index=False)
            pd.DataFrame(histories).to_csv(history_path, index=False)
            logger.info("candidate=%d seed=%d validation_score=%.4f", candidate_id, seed, best["validation_selection_score"])
            if device == "mps":
                torch.mps.empty_cache()
    frame = pd.DataFrame(rows)
    expected = len(candidates) * len(seeds)
    if len(frame) != expected:
        raise RuntimeError(f"Tuning incomplete: expected {expected} rows, observed {len(frame)}")
    summary = (
        frame.groupby(["focal_mix", "rare_sampler_power", "contrastive_weight"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            validation_selection_score=("validation_selection_score", "mean"),
            validation_f1_macro=("validation_f1_macro", "mean"),
            validation_rare_f1_macro=("validation_rare_f1_macro", "mean"),
        )
        .sort_values(["validation_selection_score", "validation_f1_macro"], ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "candidate_id", np.arange(len(summary), dtype=int))
    summary.to_csv(root / "results" / "validation_tuning_summary.csv", index=False)
    selected = summary.iloc[0].to_dict()
    selected.update(
        {
            "dataset_role": "development_only",
            "selection_rule": "highest mean of macro-F1 and rare-class macro-F1 on validation only",
            "test_access_count": 0,
        }
    )
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "SELECTED_HYPERPARAMETERS.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    logger.info("selected candidate=%d validation_score=%.4f", int(selected["candidate_id"]), selected["validation_selection_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
