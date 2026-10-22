from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from rareguard.config import load_yaml
from rareguard.utils.logging import configure_logging
from rareguard.utils.provenance import write_provenance_manifest


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No results are available."
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


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


def full_beats_ablations(central: pd.DataFrame) -> tuple[bool, list[dict[str, object]]]:
    ablations = sorted(model for model in central.get("model", pd.Series(dtype=str)).unique() if str(model).startswith("RareGuard-no-"))
    comparisons: list[dict[str, object]] = []
    if central.empty or not ablations:
        return False, comparisons
    for dataset in sorted(central["dataset"].unique()):
        full = central[(central["dataset"] == dataset) & (central["model"] == "RareGuard-XIDS")].set_index("seed")
        for ablation in ablations:
            reduced = central[(central["dataset"] == dataset) & (central["model"] == ablation)].set_index("seed")
            common = sorted(set(full.index) & set(reduced.index))
            for metric in ("f1_macro", "rare_f1_macro"):
                gain = float((full.loc[common, metric] - reduced.loc[common, metric]).mean()) if common else float("nan")
                comparisons.append(
                    {
                        "dataset": dataset,
                        "ablation": ablation,
                        "metric": metric,
                        "paired_seeds": len(common),
                        "mean_gain_full_minus_ablation": gain,
                    }
                )
    frame = pd.DataFrame(comparisons)
    passed = bool(not frame.empty and frame["mean_gain_full_minus_ablation"].notna().all() and (frame["mean_gain_full_minus_ablation"] > 0).all())
    return passed, comparisons


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    valid = sorted((value, index) for index, value in enumerate(p_values) if np.isfinite(value))
    running = 0.0
    total = len(valid)
    for rank, (value, index) in enumerate(valid):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def paired_statistical_tests(central: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if central.empty:
        return pd.DataFrame()
    for dataset in sorted(central["dataset"].unique()):
        base = central[(central["dataset"] == dataset) & (central["model"] == "NOVA-XIDS")].set_index("seed")
        full = central[(central["dataset"] == dataset) & (central["model"] == "RareGuard-XIDS")].set_index("seed")
        common = sorted(set(base.index) & set(full.index))
        for metric in ("f1_macro", "rare_f1_macro", "balanced_accuracy", "mcc", "ece_15"):
            differences = (full.loc[common, metric] - base.loc[common, metric]).dropna().to_numpy(dtype=float) if common else np.asarray([])
            if len(differences) and not np.allclose(differences, 0.0):
                p_value = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
            elif len(differences):
                p_value = 1.0
            else:
                p_value = float("nan")
            standard_deviation = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
            effect = float(np.mean(differences) / standard_deviation) if np.isfinite(standard_deviation) and standard_deviation > 0 else float("nan")
            nonzero = differences[~np.isclose(differences, 0.0)]
            rank_biserial = float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero)) if len(nonzero) else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "paired_seeds": len(differences),
                    "mean_difference_rareguard_minus_nova": float(np.mean(differences)) if len(differences) else float("nan"),
                    "paired_cohen_dz": effect,
                    "paired_sign_effect": rank_biserial,
                    "wilcoxon_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["holm_adjusted_p"] = holm_adjust(frame["wilcoxon_p"].tolist())
    return frame


def continual_superiority(continual: pd.DataFrame) -> tuple[bool, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    if continual.empty:
        return False, rows
    for dataset in sorted(continual["dataset"].unique()):
        subset = continual[continual["dataset"] == dataset]
        full = subset[subset["method"] == "RareGuard-Continual"]
        for baseline in ("RareGuard-FineTune", "RareGuard-ReservoirReplay"):
            reduced = subset[subset["method"] == baseline]
            for metric, direction in (("rare_f1_macro", "higher"), ("mean_validation_forgetting", "lower")):
                full_mean = float(full[metric].mean()) if not full.empty else float("nan")
                baseline_mean = float(reduced[metric].mean()) if not reduced.empty else float("nan")
                gain = full_mean - baseline_mean
                passed = gain > 0 if direction == "higher" else gain < 0
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "metric": metric,
                        "direction": direction,
                        "rareguard_continual_mean": full_mean,
                        "baseline_mean": baseline_mean,
                        "difference": gain,
                        "passed": bool(passed),
                    }
                )
    return bool(rows and all(row["passed"] for row in rows)), rows


def agent_policy_value(agentic: pd.DataFrame) -> tuple[bool, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    if agentic.empty:
        return False, rows
    for dataset in sorted(agentic["dataset"].unique()):
        subset = agentic[agentic["dataset"] == dataset]
        agent = subset[subset["policy"] == "rareguard_agent"]
        light = subset[subset["policy"] == "always_light_nova"]
        if agent.empty or light.empty:
            rows.append({"dataset": dataset, "metric": "required_policies", "passed": False})
            continue
        f1_gain = float(agent["f1_macro"].mean() - light["f1_macro"].mean())
        rare_gain = float(agent["rare_f1_macro"].mean() - light["rare_f1_macro"].mean())
        heavy_ratio = float(agent["heavy_call_ratio"].mean())
        rows.extend(
            [
                {"dataset": dataset, "metric": "f1_macro_gain_over_light", "value": f1_gain, "passed": f1_gain > 0},
                {"dataset": dataset, "metric": "rare_f1_gain_over_light", "value": rare_gain, "passed": rare_gain > 0},
                {"dataset": dataset, "metric": "heavy_call_ratio", "value": heavy_ratio, "passed": 0 < heavy_ratio < 1},
            ]
        )
    return bool(rows and all(bool(row["passed"]) for row in rows)), rows


def make_figures(
    root: Path,
    central: pd.DataFrame,
    continual: pd.DataFrame,
    federated: pd.DataFrame,
    agentic: pd.DataFrame,
) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if central.empty or not {"model", "dataset"}.issubset(central.columns):
        return
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
    if not continual.empty:
        summary = continual.groupby("method").agg(
            rare_f1_macro=("rare_f1_macro", "mean"),
            forgetting=("mean_validation_forgetting", "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(summary["forgetting"], summary["rare_f1_macro"], s=80, color="#8a5a2b")
        for _, row in summary.iterrows():
            ax.annotate(row["method"], (row["forgetting"], row["rare_f1_macro"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Mean validation forgetting (lower is better)")
        ax.set_ylabel("Final rare-class macro-F1")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "continual_memory_tradeoff.png", dpi=300)
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
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    central_path = result_dir / "central_seed_metrics.csv"
    baseline_path = result_dir / "baseline_seed_metrics.csv"
    tuning_path = result_dir / "validation_tuning_candidates.csv"
    continual_path = result_dir / "continual_seed_metrics.csv"
    federated_path = result_dir / "federated_seed_metrics.csv"
    agentic_path = result_dir / "agentic_policy_metrics.csv"
    central = pd.read_csv(central_path) if central_path.exists() else pd.DataFrame()
    baselines = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    tuning = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()
    continual = pd.read_csv(continual_path) if continual_path.exists() else pd.DataFrame()
    federated = pd.read_csv(federated_path) if federated_path.exists() else pd.DataFrame()
    agentic = pd.read_csv(agentic_path) if agentic_path.exists() else pd.DataFrame()
    datasets = list(load_yaml(args.datasets)["datasets"])
    development_datasets = set(config["protocol"].get("development_datasets", []))
    confirmation_datasets = set(config["protocol"].get("confirmation_datasets", []))
    seeds = [int(value) for value in config["project"]["seeds"]]
    ablation_seeds = [int(value) for value in config["protocol"].get("ablation_seeds", [])]
    federated_seeds = [int(value) for value in config["federated"].get("seeds", seeds)]
    continual_seeds = [int(value) for value in config["continual"].get("seeds", seeds)]
    expected_central = len(datasets) * (len(seeds) * 2 + len(ablation_seeds) * 3)
    expected_baselines = len(datasets) * len(seeds) * 3
    expected_continual = len(datasets) * len(continual_seeds) * 3
    expected_federated = len(datasets) * len(federated_seeds) * 4
    expected_agentic = len(datasets) * len(seeds) * 3
    expected_tuning = (
        len(config["tuning"]["seeds"])
        * len(config["tuning"]["focal_mix"])
        * len(config["tuning"]["rare_sampler_power"])
        * len(config["tuning"]["contrastive_weight"])
    )
    data_checks = []
    acquisition_modes: set[str] = set()
    for dataset in datasets:
        path = root / "processed" / dataset / "split_metadata.json"
        acquisition_path = root / "processed" / dataset / "acquisition_metadata.json"
        ready = False
        detail = "missing"
        evidence = "missing"
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            ready = bool(metadata.get("ready")) and metadata.get("transform_fit_scope") == "training_only"
            detail = str(metadata.get("duplicate_hash_overlap"))
        if acquisition_path.exists():
            evidence = str(json.loads(acquisition_path.read_text(encoding="utf-8")).get("evidence", "unknown"))
            acquisition_modes.add(evidence)
        data_checks.append({"dataset": dataset, "ready": ready, "detail": detail, "evidence": evidence})
    gains = paired_gains(central) if not central.empty else pd.DataFrame()
    gains.to_csv(result_dir / "paired_gain_ci.csv", index=False)
    statistical_tests = paired_statistical_tests(central)
    statistical_tests.to_csv(result_dir / "paired_statistical_tests.csv", index=False)
    main_gains = gains[
        gains["metric"].isin(["f1_macro", "rare_f1_macro"])
        & gains["dataset"].isin(confirmation_datasets)
    ] if not gains.empty else gains
    superiority = bool(not main_gains.empty and (main_gains["mean_gain_rareguard_minus_nova"] > 0).all())
    confirmation_central = central[central["dataset"].isin(confirmation_datasets)] if not central.empty else central
    ablation_superiority, ablation_comparisons = full_beats_ablations(confirmation_central)
    pd.DataFrame(ablation_comparisons).to_csv(result_dir / "ablation_paired_gains.csv", index=False)
    confirmation_continual = continual[continual["dataset"].isin(confirmation_datasets)] if not continual.empty else continual
    continual_better, continual_comparisons = continual_superiority(confirmation_continual)
    pd.DataFrame(continual_comparisons).to_csv(result_dir / "continual_comparisons.csv", index=False)
    confirmation_agentic = agentic[agentic["dataset"].isin(confirmation_datasets)] if not agentic.empty else agentic
    agent_better, agent_comparisons = agent_policy_value(confirmation_agentic)
    pd.DataFrame(agent_comparisons).to_csv(result_dir / "agent_policy_comparisons.csv", index=False)
    all_acquisitions_declared = len([item for item in data_checks if item["evidence"] != "missing"]) == len(datasets)
    range_sample_only = all_acquisitions_declared and acquisition_modes == {"remote_range_sample_from_canonical_drive_dataset"}
    full_corpus_reproduction = all_acquisitions_declared and all(
        item["evidence"] == "full_corpus_from_canonical_drive_dataset" for item in data_checks
    )
    selected_path = root / "reports" / "SELECTED_HYPERPARAMETERS.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    tuning_validation_only = bool(
        len(tuning) == expected_tuning
        and not tuning.empty
        and (tuning["test_access_count"] == 0).all()
        and set(tuning["dataset"].astype(str)).issubset(development_datasets)
    )
    selected_hyperparameters_frozen = bool(
        selected
        and np.isclose(float(selected["focal_mix"]), float(config["training"]["focal_mix"]))
        and np.isclose(float(selected["rare_sampler_power"]), float(config["training"]["rare_sampler_power"]))
        and np.isclose(float(selected["contrastive_weight"]), float(config["training"]["contrastive_weight"]))
    )
    gates = {
        "all_data_splits_ready": all(item["ready"] for item in data_checks),
        "validation_only_tuning_complete": tuning_validation_only,
        "selected_hyperparameters_frozen": selected_hyperparameters_frozen,
        "central_complete": len(central) == expected_central,
        "classical_baselines_complete": len(baselines) == expected_baselines,
        "continual_complete": len(continual) == expected_continual,
        "federated_complete": len(federated) == expected_federated,
        "agentic_complete": len(agentic) == expected_agentic,
        "locked_test_once": bool(
            not central.empty
            and (central["test_access_count"] == 1).all()
            and not continual.empty
            and (continual["test_access_count"] == 1).all()
            and (federated.empty or (federated["test_access_count"] == 1).all())
        ),
        "confirmation_rareguard_mean_superiority_macro_and_rare": superiority,
        "full_model_beats_all_ablations_macro_and_rare": ablation_superiority,
        "continual_memory_beats_replay_controls": continual_better,
        "agent_improves_over_light_with_partial_escalation": agent_better,
        "joule_energy_claim_supported": False,
        "full_corpus_reproduction": full_corpus_reproduction,
    }
    core_gates = {
        key: value
        for key, value in gates.items()
        if key not in {"joule_energy_claim_supported", "full_corpus_reproduction"}
    }
    if all(core_gates.values()):
        status = "RANGE_SAMPLE_READY_WITHOUT_ENERGY_CLAIM" if range_sample_only else "READY_WITHOUT_ENERGY_CLAIM"
    else:
        status = "NOT_READY"
    readiness = {
        "status": status,
        "gates": gates,
        "expected_rows": {"tuning": expected_tuning, "baselines": expected_baselines, "central": expected_central, "continual": expected_continual, "federated": expected_federated, "agentic": expected_agentic},
        "observed_rows": {"tuning": len(tuning), "baselines": len(baselines), "central": len(central), "continual": len(continual), "federated": len(federated), "agentic": len(agentic)},
        "dataset_roles": {"development": sorted(development_datasets), "external_confirmation": sorted(confirmation_datasets)},
        "data_checks": data_checks,
        "evidence_boundary": (
            "Current experiments use bounded deterministic range samples across canonical Drive CSV files, not every source row. "
            "No full-corpus claim is permitted. No energy-in-joules claim is permitted without an external hardware power meter."
        ),
    }
    (report_dir / "PUBLICATION_READINESS.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    sections = [
        "# RareGuard-XIDS Publication Evidence Audit",
        f"Status: **{readiness['status']}**",
        "## Readiness Gates",
        markdown_table(pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()])),
        "## Paired Seed Gains",
        markdown_table(gains) if not gains.empty else "Central experiments are incomplete.",
        "## Paired Statistical Tests",
        markdown_table(statistical_tests) if not statistical_tests.empty else "Central experiments are incomplete.",
        "## Evidence Boundary",
        readiness["evidence_boundary"],
        "Smoke outputs under `artifacts/smoke` are pipeline evidence only and are excluded from publication claims.",
    ]
    (report_dir / "PUBLICATION_EVIDENCE_AUDIT.md").write_text("\n\n".join(sections), encoding="utf-8")
    make_figures(root, central, continual, federated, agentic)
    write_provenance_manifest(Path(__file__).resolve().parents[3], root)
    logger.info("Publication readiness status=%s", readiness["status"])
    return 0 if readiness["status"] != "NOT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
