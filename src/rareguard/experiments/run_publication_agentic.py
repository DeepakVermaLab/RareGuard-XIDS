from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rareguard.config import load_yaml
from rareguard.evaluation.metrics import classification_metrics
from rareguard.utils.logging import configure_logging


def drift_scores(probabilities: np.ndarray, decay: float = 0.99) -> np.ndarray:
    reference = probabilities[0].copy()
    scores = np.zeros(len(probabilities), dtype=float)
    for index, current in enumerate(probabilities):
        scores[index] = 0.5 * float(np.abs(current - reference).sum())
        reference = decay * reference + (1.0 - decay) * current
    return np.clip(scores, 0.0, 1.0)


def policy_mask(
    probabilities: np.ndarray,
    rare_ids: set[int],
    uncertainty_threshold: float,
    rare_probability_threshold: float,
    drift_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uncertainty = 1.0 - probabilities.max(axis=1)
    rare_probability = probabilities[:, sorted(rare_ids)].max(axis=1) if rare_ids else np.zeros(len(probabilities))
    drift = drift_scores(probabilities)
    mask = (uncertainty >= uncertainty_threshold) | (rare_probability >= rare_probability_threshold) | (drift >= drift_threshold)
    return mask, uncertainty, rare_probability, drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a measured NOVA-to-RareGuard 24/7 escalation policy.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_agentic.log")
    central = pd.read_csv(root / "results" / "central_seed_metrics.csv")
    keys = sorted(central["dataset"].unique()) if args.dataset == "all" else [args.dataset]
    settings = config["agent"]
    rows: list[dict[str, object]] = []
    decisions: list[pd.DataFrame] = []
    for dataset in keys:
        metadata = json.loads((root / "processed" / dataset / "split_metadata.json").read_text(encoding="utf-8"))
        counts = np.asarray(metadata["split_class_counts"]["train"])
        cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * counts.sum()))
        rare_ids = {int(i) for i, count in enumerate(counts) if count <= cutoff}
        for seed in sorted(central.loc[central["dataset"] == dataset, "seed"].unique()):
            subset = central[(central["dataset"] == dataset) & (central["seed"] == seed)]
            nova_row = subset[subset["model"] == "NOVA-XIDS"].iloc[0]
            rare_row = subset[subset["model"] == "RareGuard-XIDS"].iloc[0]
            nova = np.load(nova_row["prediction_path"])
            rare = np.load(rare_row["prediction_path"])
            y = nova["y"]
            nova_probabilities = nova["probabilities"]
            rare_probabilities = rare["probabilities"]
            mask, uncertainty, rare_probability, drift = policy_mask(
                nova_probabilities,
                rare_ids,
                float(settings["uncertainty_threshold"]),
                float(settings["rare_probability_threshold"]),
                float(settings["drift_threshold"]),
            )
            policies = {
                "always_light_nova": nova_probabilities,
                "always_heavy_rareguard": rare_probabilities,
                "rareguard_agent": np.where(mask[:, None], rare_probabilities, nova_probabilities),
            }
            nova_seconds = float(nova_row["test_seconds"])
            rare_seconds = float(rare_row["test_seconds"])
            light_per_sample = nova_seconds / len(y)
            heavy_per_sample = rare_seconds / len(y)
            for policy, probabilities in policies.items():
                heavy_calls = len(y) if policy == "always_heavy_rareguard" else (int(mask.sum()) if policy == "rareguard_agent" else 0)
                measured_time_projection = len(y) * light_per_sample + heavy_calls * heavy_per_sample
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "policy": policy,
                        "events": len(y),
                        "heavy_calls": heavy_calls,
                        "heavy_call_ratio": float(heavy_calls / len(y)),
                        "projected_inference_seconds_from_measured_runs": measured_time_projection,
                        "projected_events_per_second": float(len(y) / max(measured_time_projection, 1e-9)),
                        "energy_evidence": "not_measured_no_joule_claim",
                        **classification_metrics(y, probabilities, rare_ids),
                    }
                )
            decisions.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "index": np.arange(len(y)),
                        "uncertainty": uncertainty,
                        "rare_probability": rare_probability,
                        "drift_score": drift,
                        "wake_heavy_detector": mask,
                        "true_class": y,
                        "light_prediction": nova_probabilities.argmax(axis=1),
                        "heavy_prediction": rare_probabilities.argmax(axis=1),
                    }
                )
            )
            logger.info("%s seed=%d heavy_ratio=%.4f", dataset, seed, mask.mean())
    result_dir = root / "results"
    pd.DataFrame(rows).to_csv(result_dir / "agentic_policy_metrics.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(result_dir / "agentic_decisions.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
