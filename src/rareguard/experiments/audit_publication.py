from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rareguard.config import load_yaml
from rareguard.utils.logging import configure_logging


def paired_gains(central: pd.DataFrame) -> pd.DataFrame:
    base = central[central["model"] == "NOVA-XIDS"].set_index(["dataset", "seed"])
    full = central[central["model"] == "RareGuard-XIDS"].set_index(["dataset", "seed"])
    rows = []
    for dataset in sorted(set(base.index.get_level_values(0)) & set(full.index.get_level_values(0))):
        for metric in ["f1_macro", "rare_f1_macro", "balanced_accuracy", "mcc", "ece_15"]:
            values = (full.loc[dataset, metric] - base.loc[dataset, metric]).dropna().astype(float)
            mean = float(values.mean()) if len(values) else float("nan")
            half = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "paired_seeds": len(values),
                    "mean_gain_rareguard_minus_nova": mean,
                    "ci95_lower": mean - half,
                    "ci95_upper": mean + half,
                }
            )
    return pd.DataFrame(rows)


def make_figures(root: Path, central: pd.DataFrame, federated: pd.DataFrame, agentic: pd.DataFrame) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    main = central[central["model"].isin(["NOVA-XIDS", "RareGuard-XIDS"])]
    for metric, filename, label in [
        ("f1_macro", "central_macro_f1_ci.png", "Macro-F1"),
        ("rare_f1_macro", "central_rare_f1_ci.png", "Rare-class macro-F1"),
    ]:
        summary = main.groupby(["dataset", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
        summary["half"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
        pivot = summary.pivot(index="dataset", columns="model", values="mean")
        errors = summary.pivot(index="dataset", columns="model", values="half").reindex_like(pivot)
        ax = pivot.plot(kind="bar", yerr=errors, capsize=3, figsize=(9, 4.8), color=["#3b6c8e", "#b64c3c"])
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=300)
        plt.close()
    if not federated.empty:
        summary = federated.groupby("method").agg(f1_macro=("f1_macro", "mean"), communication_mb=("communication_bytes", lambda x: x.mean() / 1e6)).reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for _, row in summary.iterrows():
            ax.scatter(row["communication_mb"], row["f1_macro"], s=70)
            ax.annotate(row["method"], (row["communication_mb"], row["f1_macro"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Mean communication (MB)")
        ax.set_ylabel("Mean macro-F1")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "federated_communication_tradeoff.png", dpi=300)
        plt.close()
    if not agentic.empty:
        summary = agentic.groupby("policy").agg(f1_macro=("f1_macro", "mean"), heavy_ratio=("heavy_call_ratio", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(summary["heavy_ratio"], summary["f1_macro"], s=80, color="#2f7d62")
        for _, row in summary.iterrows():
            ax.annotate(row["policy"], (row["heavy_ratio"], row["f1_macro"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Heavy-model invocation ratio")
        ax.set_ylabel("Mean macro-F1")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "agentic_accuracy_compute_tradeoff.png", dpi=300)
        plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit publication evidence without altering experimental results.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "audit_publication.log")
    result_dir = root / "results"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    central_path = result_dir / "central_seed_metrics.csv"
    federated_path = result_dir / "federated_seed_metrics.csv"
    agentic_path = result_dir / "agentic_policy_metrics.csv"
    central = pd.read_csv(central_path) if central_path.exists() else pd.DataFrame()
    federated = pd.read_csv(federated_path) if federated_path.exists() else pd.DataFrame()
    agentic = pd.read_csv(agentic_path) if agentic_path.exists() else pd.DataFrame()
    datasets = list(load_yaml(args.datasets)["datasets"])
    seeds = [int(value) for value in config["project"]["seeds"]]
    ablation_seeds = [int(value) for value in config["protocol"].get("ablation_seeds", [])]
    federated_seeds = [int(value) for value in config["federated"].get("seeds", seeds)]
    expected_central = len(datasets) * (len(seeds) * 2 + len(ablation_seeds) * 3)
    expected_federated = len(datasets) * len(federated_seeds) * 4
    expected_agentic = len(datasets) * len(seeds) * 3
    data_checks = []
    for dataset in datasets:
        path = root / "processed" / dataset / "split_metadata.json"
        ready = False
        detail = "missing"
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            ready = bool(metadata.get("ready")) and metadata.get("transform_fit_scope") == "training_only"
            detail = str(metadata.get("duplicate_hash_overlap"))
        data_checks.append({"dataset": dataset, "ready": ready, "detail": detail})
    gains = paired_gains(central) if not central.empty else pd.DataFrame()
    gains.to_csv(result_dir / "paired_gain_ci.csv", index=False)
    main_gains = gains[gains["metric"].isin(["f1_macro", "rare_f1_macro"])] if not gains.empty else gains
    superiority = bool(not main_gains.empty and (main_gains["mean_gain_rareguard_minus_nova"] > 0).all())
    gates = {
        "all_data_splits_ready": all(item["ready"] for item in data_checks),
        "central_complete": len(central) == expected_central,
        "federated_complete": len(federated) == expected_federated,
        "agentic_complete": len(agentic) == expected_agentic,
        "locked_test_once": bool(not central.empty and (central["test_access_count"] == 1).all() and (federated.empty or (federated["test_access_count"] == 1).all())),
        "rareguard_mean_superiority_macro_and_rare": superiority,
        "joule_energy_claim_supported": False,
    }
    readiness = {
        "status": "READY_WITHOUT_ENERGY_CLAIM" if all(value for key, value in gates.items() if key != "joule_energy_claim_supported") else "NOT_READY",
        "gates": gates,
        "expected_rows": {"central": expected_central, "federated": expected_federated, "agentic": expected_agentic},
        "observed_rows": {"central": len(central), "federated": len(federated), "agentic": len(agentic)},
        "data_checks": data_checks,
        "evidence_boundary": "No energy-in-joules claim is permitted without an external hardware power meter.",
    }
    (report_dir / "PUBLICATION_READINESS.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    sections = [
        "# RareGuard-XIDS Publication Evidence Audit",
        f"Status: **{readiness['status']}**",
        "## Readiness Gates",
        pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_markdown(index=False),
        "## Paired Seed Gains",
        gains.to_markdown(index=False) if not gains.empty else "Central experiments are incomplete.",
        "## Evidence Boundary",
        readiness["evidence_boundary"],
        "Smoke outputs under `artifacts/smoke` are pipeline evidence only and are excluded from publication claims.",
    ]
    (report_dir / "PUBLICATION_EVIDENCE_AUDIT.md").write_text("\n\n".join(sections), encoding="utf-8")
    make_figures(root, central, federated, agentic)
    logger.info("Publication readiness status=%s", readiness["status"])
    return 0 if readiness["status"] != "NOT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
