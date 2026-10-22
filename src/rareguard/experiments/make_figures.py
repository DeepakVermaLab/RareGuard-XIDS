from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rareguard.config import load_config
from rareguard.utils.logging import configure_logging


def save_grouped_bar(frame: pd.DataFrame, x_col: str, hue_col: str, y_col: str, title: str, ylabel: str, out: Path) -> None:
    if frame.empty or not {x_col, hue_col, y_col}.issubset(frame.columns):
        return
    pivot = frame.pivot_table(index=x_col, columns=hue_col, values=y_col, aggfunc="max").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 1.4), 4.8), width=0.78)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(title="")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close()


def save_single_bar(frame: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out: Path) -> None:
    if frame.empty or not {x_col, y_col}.issubset(frame.columns):
        return
    ordered = frame.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(max(7, len(ordered) * 1.3), 4.5))
    ax.bar(ordered[x_col], ordered[y_col], color="#2f6f73")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close()


def read_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-support figures from RareGuard-XIDS results.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = configure_logging(cfg.artifact_root / "logs" / "make_figures.log")
    result_dir = cfg.artifact_root / "results"
    figure_dir = cfg.artifact_root / "figures"

    save_grouped_bar(
        read_optional(result_dir / "baseline_metrics.csv"),
        "dataset",
        "model",
        "f1_macro",
        "Classical baseline macro-F1",
        "Macro-F1",
        figure_dir / "baseline_macro_f1.png",
    )
    save_grouped_bar(
        read_optional(result_dir / "embedding_metrics.csv"),
        "dataset",
        "model",
        "f1_macro",
        "FALCON-ID vs RareGuard embedding macro-F1",
        "Macro-F1",
        figure_dir / "embedding_macro_f1.png",
    )
    save_single_bar(
        read_optional(result_dir / "agentic_energy_summary.csv"),
        "dataset",
        "estimated_energy_reduction_ratio",
        "Agentic monitoring estimated energy reduction",
        "Reduction ratio",
        figure_dir / "agentic_energy_reduction.png",
    )
    logger.info("Wrote figures under %s", figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
