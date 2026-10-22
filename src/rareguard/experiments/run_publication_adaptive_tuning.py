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
from rareguard.experiments.adaptive_config import selected_hyperparameters_path
from rareguard.experiments.run_publication_central import load_splits, resolve_device
from rareguard.experiments.run_publication_tuning import train_validation_candidate
from rareguard.utils.logging import configure_logging


def candidate_grid(config: dict) -> list[dict[str, float]]:
    tuning = config["adaptive_tuning"]
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
    parser = argparse.ArgumentParser(
        description="Run the prespecified per-dataset validation-only RareGuard adaptation procedure."
    )
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_adaptive_tuning.log")
    device = resolve_device(args.device or config["adaptive_tuning"].get("device", "cpu"))
    dataset_keys = list(load_yaml(args.datasets)["datasets"])
    keys = dataset_keys if args.dataset == "all" else [args.dataset]
    unknown = sorted(set(keys) - set(dataset_keys))
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}")

    result_path = root / "results" / "adaptive_tuning_candidates.csv"
    history_path = root / "results" / "adaptive_tuning_history.csv"
    summary_path = root / "results" / "adaptive_tuning_summary.csv"
    rows = pd.read_csv(result_path).to_dict("records") if args.resume and result_path.exists() else []
    histories = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    complete = {
        (
            str(row["dataset"]),
            float(row["focal_mix"]),
            float(row["rare_sampler_power"]),
            float(row["contrastive_weight"]),
            int(row["seed"]),
        )
        for row in rows
    }
    seeds = [int(value) for value in config["adaptive_tuning"]["seeds"]]
    candidates = candidate_grid(config)
    for dataset in keys:
        arrays = load_splits(root / "processed" / dataset / "splits.npz")
        for candidate_id, candidate in enumerate(candidates):
            for seed in seeds:
                key = (
                    dataset,
                    candidate["focal_mix"],
                    candidate["rare_sampler_power"],
                    candidate["contrastive_weight"],
                    seed,
                )
                if key in complete:
                    continue
                candidate_config = copy.deepcopy(config)
                candidate_config["training"].update(candidate)
                logger.info("dataset=%s candidate=%d seed=%d settings=%s", dataset, candidate_id, seed, candidate)
                best, history = train_validation_candidate(arrays, candidate_config, seed, device)
                rows.append(
                    {
                        "dataset": dataset,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "device": device,
                        **candidate,
                        **best,
                        "test_access_count": 0,
                    }
                )
                histories.extend(
                    {
                        "dataset": dataset,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        **candidate,
                        **item,
                    }
                    for item in history
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                pd.DataFrame(histories).to_csv(history_path, index=False)
                if device == "mps":
                    torch.mps.empty_cache()

        frame = pd.DataFrame(rows)
        subset = frame[frame["dataset"] == dataset].copy()
        expected = len(candidates) * len(seeds)
        if len(subset) != expected:
            raise RuntimeError(f"Adaptive tuning incomplete for {dataset}: expected {expected}, observed {len(subset)}")
        summary = (
            subset.groupby(["dataset", "focal_mix", "rare_sampler_power", "contrastive_weight"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                validation_selection_score=("validation_selection_score", "mean"),
                validation_f1_macro=("validation_f1_macro", "mean"),
                validation_rare_f1_macro=("validation_rare_f1_macro", "mean"),
            )
            .sort_values(
                ["validation_selection_score", "validation_rare_f1_macro", "validation_f1_macro"],
                ascending=False,
            )
            .reset_index(drop=True)
        )
        existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        if not existing_summary.empty:
            existing_summary = existing_summary[existing_summary["dataset"] != dataset]
        combined_summary = pd.concat([existing_summary, summary], ignore_index=True)
        combined_summary.to_csv(summary_path, index=False)
        selected = summary.iloc[0].to_dict()
        selected.update(
            {
                "candidate_grid_size": len(candidates),
                "selection_seed_count": len(seeds),
                "selection_rule": "highest mean validation composite; tie-break rare-class macro-F1 then macro-F1",
                "validation_composite": "0.5 * (macro_f1 + rare_class_macro_f1)",
                "test_access_count": 0,
            }
        )
        selected_path = selected_hyperparameters_path(root, dataset)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            json.dumps(
                selected,
                indent=2,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ),
            encoding="utf-8",
        )
        logger.info(
            "selected dataset=%s validation_score=%.4f settings=(%.3f, %.3f, %.3f)",
            dataset,
            selected["validation_selection_score"],
            selected["focal_mix"],
            selected["rare_sampler_power"],
            selected["contrastive_weight"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
