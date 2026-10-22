from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rareguard.config import load_yaml
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.adaptive_config import apply_dataset_hyperparameters
from rareguard.experiments.run_publication_central import (
    Variant,
    apply_temperature,
    build_model,
    infer,
    load_splits,
    resolve_device,
)
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


def perturb_features(x: np.ndarray, kind: str, strength: float, seed: int, clip_value: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = x.copy()
    if kind == "feature_mask":
        result[rng.random(result.shape) < strength] = 0.0
    elif kind == "gaussian_noise":
        result += rng.normal(0.0, strength, size=result.shape).astype(result.dtype)
    else:
        raise ValueError(f"Unknown robustness perturbation: {kind}")
    return np.clip(result, -clip_value, clip_value)


def scenarios(config: dict) -> list[tuple[str, float]]:
    result = []
    for value in config["robustness"]["feature_mask_fraction"]:
        result.append(("feature_mask", float(value)))
    for value in config["robustness"]["gaussian_noise_std"]:
        result.append(("gaussian_noise", float(value)))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic RareGuard robustness stress tests.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_robustness.log")
    device = resolve_device(args.device or config["project"].get("device", "cpu"))
    dataset_keys = list(load_yaml(args.datasets)["datasets"])
    keys = dataset_keys if args.dataset == "all" else [args.dataset]
    central = pd.read_csv(root / "results" / "central_seed_metrics.csv")
    result_path = root / "results" / "robustness_seed_metrics.csv"
    rows = pd.read_csv(result_path).to_dict("records") if args.resume and result_path.exists() else []
    complete = {
        (str(row["dataset"]), int(row["seed"]), str(row["model"]), str(row["perturbation"]), float(row["strength"]))
        for row in rows
    }
    clip_value = float(config["data"]["feature_clip_value"])
    for dataset in keys:
        dataset_config = apply_dataset_hyperparameters(config, root, dataset)
        arrays = load_splits(root / "processed" / dataset / "splits.npz")
        classes = int(max(arrays["y_train"].max(), arrays["y_test"].max())) + 1
        counts = np.bincount(arrays["y_train"], minlength=classes)
        cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * len(arrays["y_train"])))
        rare_ids = {int(index) for index, count in enumerate(counts) if count <= cutoff}
        for seed in [int(value) for value in config["project"]["seeds"]]:
            set_seed(seed)
            central_seed = central[(central["dataset"] == dataset) & (central["seed"] == seed)]
            for scenario_id, (kind, strength) in enumerate(scenarios(config)):
                stressed = perturb_features(
                    arrays["X_test"], kind, strength, seed * 100 + scenario_id, clip_value
                )
                for label, variant in (
                    ("NOVA-XIDS", Variant("NOVA-XIDS", False)),
                    ("RareGuard-XIDS", Variant("RareGuard-XIDS", True)),
                ):
                    key = (dataset, seed, label, kind, strength)
                    if key in complete:
                        continue
                    central_row = central_seed[central_seed["model"] == label]
                    if len(central_row) != 1:
                        raise RuntimeError(f"Missing unique central model for {dataset} seed={seed} {label}")
                    row = central_row.iloc[0]
                    model = build_model(variant, arrays["X_train"].shape[1], classes, dataset_config).to(device)
                    model.load_state_dict(torch.load(row["model_path"], map_location=device, weights_only=True))
                    raw_probabilities, _ = infer(
                        model,
                        stressed,
                        arrays["y_test"],
                        int(dataset_config["training"]["batch_size"]),
                        device,
                        variant.rareguard,
                    )
                    probabilities = apply_temperature(raw_probabilities, float(row["temperature"]))
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model": label,
                            "perturbation": kind,
                            "strength": strength,
                            "device": device,
                            "test_access_count": 1,
                            **classification_metrics(arrays["y_test"], probabilities, rare_ids),
                        }
                    )
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(result_path, index=False)
                    del model
                    gc.collect()
        logger.info("Completed robustness scenarios for %s", dataset)
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["dataset", "model", "perturbation", "strength"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            rare_f1_macro_mean=("rare_f1_macro", "mean"),
            rare_f1_macro_std=("rare_f1_macro", "std"),
        )
    )
    summary.to_csv(root / "results" / "robustness_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
