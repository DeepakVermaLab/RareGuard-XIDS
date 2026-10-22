from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rareguard.config import load_config
from rareguard.utils.logging import configure_logging


def maybe_table(path: Path) -> str:
    if not path.exists():
        return f"Missing: `{path}`\n"
    frame = pd.read_csv(path)
    return frame.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = configure_logging(cfg.artifact_root / "logs" / "make_report.log")
    report_dir = cfg.artifact_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "IMPLEMENTATION_AUDIT.md"
    sections = [
        "# RareGuard-XIDS Implementation Audit",
        "## Dataset Profiles",
        maybe_table(cfg.artifact_root / "results" / "dataset_profiles.csv"),
        "## Preprocessing Summary",
        maybe_table(cfg.artifact_root / "results" / "preprocess_summary.csv"),
        "## Baseline Metrics",
        maybe_table(cfg.artifact_root / "results" / "baseline_metrics.csv"),
        "## Embedding Metrics",
        maybe_table(cfg.artifact_root / "results" / "embedding_metrics.csv"),
        "## Federated Summary",
        maybe_table(cfg.artifact_root / "results" / "federated_summary.csv"),
        "## Agentic Energy Summary",
        maybe_table(cfg.artifact_root / "results" / "agentic_energy_summary.csv"),
        "## Ablation Comparison",
        maybe_table(cfg.artifact_root / "results" / "ablation_comparison.csv"),
        "## Embedding Gain Table",
        maybe_table(cfg.artifact_root / "results" / "embedding_gain_table.csv"),
        "## Evidence Boundary",
        "Federated and agentic stages are scaffolded for CPU-first smoke testing until full model-connected runs are completed. Do not claim final paper improvements until all measured experiments, ablations, and logs are audited.",
    ]
    path.write_text("\n\n".join(sections), encoding="utf-8")
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
