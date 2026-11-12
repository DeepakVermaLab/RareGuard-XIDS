from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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


def projected_seconds(policy: str, events: int, heavy_calls: int, light_per_sample: float, heavy_per_sample: float) -> float:
    if policy == "always_light_nova":
        return events * light_per_sample
    if policy == "always_heavy_rareguard":
        return events * heavy_per_sample
    return events * light_per_sample + heavy_calls * heavy_per_sample


def adaptive_threshold_grid(values: np.ndarray, configured: list[float], quantiles: list[float]) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("Agent threshold selection received no finite validation scores.")
    candidates = [float(value) for value in configured]
    candidates.extend(float(np.quantile(finite, quantile)) for quantile in quantiles)
    candidates.append(float(np.nextafter(finite.max(), math.inf)))
    return sorted(set(candidates))


def selection_f1_metrics(y_true: np.ndarray, prediction: np.ndarray, rare_ids: set[int]) -> tuple[float, float]:
    macro = float(f1_score(y_true, prediction, average="macro", zero_division=0))
    labels = np.arange(max(int(y_true.max()), int(prediction.max())) + 1)
    per_class = f1_score(y_true, prediction, labels=labels, average=None, zero_division=0)
    rare = [class_id for class_id in sorted(rare_ids) if class_id < len(per_class)]
    rare_macro = float(per_class[rare].mean()) if rare else float("nan")
    return macro, rare_macro


def select_policy_thresholds(
    y_validation: np.ndarray,
    light_probabilities: np.ndarray,
    heavy_probabilities: np.ndarray,
    rare_ids: set[int],
    settings: dict,
) -> dict[str, object]:
    uncertainty = 1.0 - light_probabilities.max(axis=1)
    rare_probability = (
        light_probabilities[:, sorted(rare_ids)].max(axis=1)
        if rare_ids
        else np.zeros(len(light_probabilities))
    )
    drift = drift_scores(light_probabilities)
    quantiles = [float(value) for value in settings.get("adaptive_quantiles", [0.50, 0.70, 0.85, 0.95])]
    uncertainty_grid = adaptive_threshold_grid(uncertainty, settings["uncertainty_grid"], quantiles)
    rare_probability_grid = adaptive_threshold_grid(
        rare_probability,
        settings["rare_probability_grid"],
        quantiles,
    )
    drift_grid = adaptive_threshold_grid(drift, settings["drift_grid"], quantiles)
    light_prediction = light_probabilities.argmax(axis=1)
    heavy_prediction = heavy_probabilities.argmax(axis=1)
    candidates: list[dict[str, object]] = []
    for uncertainty_threshold in uncertainty_grid:
        for rare_probability_threshold in rare_probability_grid:
            for drift_threshold in drift_grid:
                mask = (
                    (uncertainty >= uncertainty_threshold)
                    | (rare_probability >= rare_probability_threshold)
                    | (drift >= drift_threshold)
                )
                heavy_ratio = float(mask.mean())
                if not (
                    float(settings["min_heavy_call_ratio"])
                    <= heavy_ratio
                    <= float(settings["max_heavy_call_ratio"])
                ):
                    continue
                prediction = np.where(mask, heavy_prediction, light_prediction)
                macro_f1, rare_f1 = selection_f1_metrics(y_validation, prediction, rare_ids)
                composite = float(macro_f1 if math.isnan(rare_f1) else 0.5 * (macro_f1 + rare_f1))
                objective = composite - float(settings["compute_penalty"]) * heavy_ratio
                candidates.append(
                    {
                        "uncertainty_threshold": float(uncertainty_threshold),
                        "rare_probability_threshold": float(rare_probability_threshold),
                        "drift_threshold": float(drift_threshold),
                        "validation_f1_macro": macro_f1,
                        "validation_rare_f1_macro": rare_f1,
                        "validation_composite": composite,
                        "validation_heavy_call_ratio": heavy_ratio,
                        "validation_selection_objective": objective,
                        "threshold_grid_source": "configured_plus_validation_quantiles",
                    }
                )
    if not candidates:
        raise RuntimeError("Agent validation grid produced no candidate inside the heavy-call ratio bounds.")
    return max(
        candidates,
        key=lambda row: (
            row["validation_selection_objective"],
            row["validation_composite"],
            -row["validation_heavy_call_ratio"],
        ),
    )


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
    selections: list[dict[str, object]] = []
    for dataset in keys:
        metadata = json.loads((root / "processed" / dataset / "split_metadata.json").read_text(encoding="utf-8"))
        counts = np.asarray(metadata["split_class_counts"]["train"])
        cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * counts.sum()))
        rare_ids = {int(i) for i, count in enumerate(counts) if count <= cutoff}
        for seed in sorted(central.loc[central["dataset"] == dataset, "seed"].unique()):
            subset = central[(central["dataset"] == dataset) & (central["seed"] == seed)]
            available_models = set(subset["model"])
            required_models = {"NOVA-XIDS", "RareGuard-XIDS"}
            if not required_models.issubset(available_models):
                logger.warning(
                    "%s seed=%s skipped because central evaluation is incomplete; missing=%s",
                    dataset,
                    seed,
                    sorted(required_models - available_models),
                )
                continue
            nova_row = subset[subset["model"] == "NOVA-XIDS"].iloc[0]
            rare_row = subset[subset["model"] == "RareGuard-XIDS"].iloc[0]
            nova = np.load(nova_row["prediction_path"])
            rare = np.load(rare_row["prediction_path"])
            y = nova["y"]
            nova_probabilities = nova["probabilities"]
            rare_probabilities = rare["probabilities"]
            if not {"validation_y", "validation_probabilities"}.issubset(nova.files) or not {
                "validation_y",
                "validation_probabilities",
            }.issubset(rare.files):
                raise RuntimeError(
                    f"Validation predictions missing for {dataset} seed={seed}; rerun central experiments."
                )
            if not np.array_equal(nova["validation_y"], rare["validation_y"]):
                raise RuntimeError(f"Validation labels differ between models for {dataset} seed={seed}")
            selection = select_policy_thresholds(
                nova["validation_y"],
                nova["validation_probabilities"],
                rare["validation_probabilities"],
                rare_ids,
                settings,
            )
            selections.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "test_access_count": 0,
                    "nova_temperature": float(nova_row["temperature"]),
                    "rareguard_temperature": float(rare_row["temperature"]),
                    **selection,
                }
            )
            mask, uncertainty, rare_probability, drift = policy_mask(
                nova_probabilities,
                rare_ids,
                selection["uncertainty_threshold"],
                selection["rare_probability_threshold"],
                selection["drift_threshold"],
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
                measured_time_projection = projected_seconds(
                    policy,
                    len(y),
                    heavy_calls,
                    light_per_sample,
                    heavy_per_sample,
                )
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
                        "test_access_count": 1,
                        **selection,
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
                        "trigger_uncertainty": uncertainty >= selection["uncertainty_threshold"],
                        "trigger_rare_signal": rare_probability >= selection["rare_probability_threshold"],
                        "trigger_drift": drift >= selection["drift_threshold"],
                        "wake_heavy_detector": mask,
                        "true_class": y,
                        "light_prediction": nova_probabilities.argmax(axis=1),
                        "heavy_prediction": rare_probabilities.argmax(axis=1),
                    }
                )
            )
            current = decisions[-1]
            current["trigger_reason"] = current.apply(
                lambda row: ";".join(
                    reason
                    for reason, active in (
                        ("uncertainty", row["trigger_uncertainty"]),
                        ("rare_attack_signal", row["trigger_rare_signal"]),
                        ("traffic_drift", row["trigger_drift"]),
                    )
                    if active
                )
                or "none",
                axis=1,
            )
            logger.info("%s seed=%d heavy_ratio=%.4f", dataset, seed, mask.mean())
    result_dir = root / "results"
    pd.DataFrame(rows).to_csv(result_dir / "agentic_policy_metrics.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(result_dir / "agentic_decisions.csv", index=False)
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(result_dir / "agentic_validation_selections.csv", index=False)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for dataset, subset in selection_frame.groupby("dataset"):
        per_seed = {str(int(row["seed"])): row for row in subset.to_dict("records")}
        deployment_default = {
            key: float(subset[key].median())
            for key in (
                "uncertainty_threshold",
                "rare_probability_threshold",
                "drift_threshold",
                "nova_temperature",
                "rareguard_temperature",
            )
        }
        payload = {
            "dataset": dataset,
            "selection_scope": "validation_only",
            "test_access_count": 0,
            "per_seed": per_seed,
            "deployment_default": deployment_default,
        }
        (report_dir / f"AGENT_POLICY_{dataset}.json").write_text(
            json.dumps(
                payload,
                indent=2,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
