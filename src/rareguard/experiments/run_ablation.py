from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rareguard.config import load_config
from rareguard.utils.logging import configure_logging


def read_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready ablation and comparison tables from completed runs.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = configure_logging(cfg.artifact_root / "logs" / "run_ablation.log")
    result_dir = cfg.artifact_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    baseline = read_optional(result_dir / "baseline_metrics.csv")
    embedding = read_optional(result_dir / "embedding_metrics.csv")
    federated = read_optional(result_dir / "federated_summary.csv")
    agentic = read_optional(result_dir / "agentic_energy_summary.csv")

    rows: list[dict[str, object]] = []
    if not baseline.empty:
        for _, row in baseline.iterrows():
            rows.append(
                {
                    "dataset": row["dataset"],
                    "stage": "classical_baseline",
                    "method": row["model"],
                    "macro_f1": row["f1_macro"],
                    "weighted_f1": row["f1_weighted"],
                    "evidence": "measured",
                }
            )
    if not embedding.empty:
        for _, row in embedding.iterrows():
            rows.append(
                {
                    "dataset": row["dataset"],
                    "stage": "neural_embedding",
                    "method": row["model"],
                    "macro_f1": row["f1_macro"],
                    "weighted_f1": row["f1_weighted"],
                    "evidence": "measured",
                }
            )
    if not federated.empty:
        for _, row in federated.iterrows():
            rows.append(
                {
                    "dataset": row["dataset"],
                    "stage": "federated",
                    "method": row["method"],
                    "macro_f1": row["f1_macro"],
                    "weighted_f1": row["f1_weighted"],
                    "evidence": row.get("evidence", "measured_federated_training"),
                }
            )

    comparison = pd.DataFrame(rows)
    comparison_path = result_dir / "ablation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    if not embedding.empty:
        pivot = embedding.pivot_table(index="dataset", columns="model", values="f1_macro", aggfunc="max").reset_index()
        if {"FalconID", "RareGuard"}.issubset(pivot.columns):
            pivot["rareguard_minus_falconid_macro_f1"] = pivot["RareGuard"] - pivot["FalconID"]
        pivot.to_csv(result_dir / "embedding_gain_table.csv", index=False)

    if not agentic.empty:
        cols = ["dataset", "total_events", "heavy_calls", "heavy_calls_saved", "estimated_energy_reduction_ratio"]
        agentic[[c for c in cols if c in agentic.columns]].to_csv(result_dir / "agentic_ablation_table.csv", index=False)

    logger.info("Wrote %s", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
