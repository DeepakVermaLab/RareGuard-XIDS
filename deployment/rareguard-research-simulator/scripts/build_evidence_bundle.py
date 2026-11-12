#!/usr/bin/env python3
"""Build a compact browser bundle from the authoritative RareGuard-XIDS v2 evidence.

The browser never invents outcomes. It receives measured aggregate tables and
contiguous windows of recorded event decisions. Any threshold what-if shown in
the UI is recomputed from those recorded scores and predictions and is labelled
as a replay, not a new benchmark run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SITE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SITE_ROOT / "evidence" / "source"
RESULTS_ROOT = SOURCE_ROOT / "results"
REPORTS_ROOT = SOURCE_ROOT / "reports"
OUTPUT_PATH = SITE_ROOT / "assets" / "data" / "evidence.json"

DATASET_LABELS = {
    "cse_cic_ids2018": "CSE-CIC-IDS2018",
    "cic_bccc_nrc_tabulariotattack_2024": "CIC-BCCC-NRC TabularIoTAttack 2024",
    "cic_iiot_2025_datasense": "CIC-IIoT 2025 DataSense",
    "edge_iiotset": "Edge-IIoTset",
    "ciciot2023": "CICIoT2023",
    "ton_iot": "TON_IoT",
}

REQUIRED_FILES = [
    RESULTS_ROOT / "agentic_decisions.csv",
    RESULTS_ROOT / "agentic_policy_metrics.csv",
    RESULTS_ROOT / "agentic_validation_selections.csv",
    RESULTS_ROOT / "central_summary_ci.csv",
    RESULTS_ROOT / "baseline_seed_metrics.csv",
    RESULTS_ROOT / "robustness_summary.csv",
    RESULTS_ROOT / "robustness_comparisons.csv",
    RESULTS_ROOT / "ablation_paired_gains.csv",
    RESULTS_ROOT / "continual_comparisons.csv",
    RESULTS_ROOT / "federated_seed_metrics.csv",
    RESULTS_ROOT / "paired_gain_ci.csv",
    RESULTS_ROOT / "paired_statistical_tests.csv",
    REPORTS_ROOT / "PUBLICATION_READINESS.json",
    REPORTS_ROOT / "PUBLICATION_EVIDENCE_AUDIT.md",
    REPORTS_ROOT / "PROVENANCE_MANIFEST.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"nan", "inf", "+inf", "-inf"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    if number.is_integer() and not any(token in lower for token in (".", "e")):
        return int(number)
    return number


def read_csv(name: str) -> list[dict[str, Any]]:
    path = RESULTS_ROOT / name
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: clean_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def compact_rows(rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> list[dict[str, Any]]:
    wanted = tuple(fields)
    return [{key: row.get(key) for key in wanted} for row in rows]


def aggregate_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["model"]))].append(row)
    metrics = ("accuracy", "f1_macro", "rare_f1_macro", "ece_15", "test_samples_per_second")
    output = []
    for (dataset, model), group in sorted(groups.items()):
        item: dict[str, Any] = {"dataset": dataset, "model": model, "seeds": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            item[metric] = mean(values) if values else None
        output.append(item)
    return output


def aggregate_federated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)
    metrics = ("f1_macro", "rare_f1_macro", "ece_15", "communication_bytes", "wall_seconds")
    output = []
    for (dataset, method), group in sorted(groups.items()):
        item: dict[str, Any] = {"dataset": dataset, "method": method, "seeds": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            item[metric] = mean(values) if values else None
        output.append(item)
    return output


def best_contiguous_window(rows: list[dict[str, Any]], window_size: int = 120) -> list[dict[str, Any]]:
    """Select a contiguous, trigger-diverse window without changing row order."""
    if len(rows) <= window_size:
        return rows
    best_start = 0
    best_score = -1.0
    step = max(1, window_size // 8)
    candidate_starts = list(range(0, len(rows) - window_size + 1, step))
    if candidate_starts[-1] != len(rows) - window_size:
        candidate_starts.append(len(rows) - window_size)
    for start in candidate_starts:
        window = rows[start : start + window_size]
        reasons = {str(row.get("trigger_reason")) for row in window}
        wakes = sum(bool(row.get("wake_heavy_detector")) for row in window)
        balance = min(wakes, window_size - wakes) / window_size
        score = len(reasons) * 10 + balance
        if score > best_score:
            best_start, best_score = start, score
    return rows[best_start : best_start + window_size]


def build_event_traces() -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    reason_counts: dict[str, int] = defaultdict(int)
    total = 0
    heavy = 0
    with (RESULTS_ROOT / "agentic_decisions.csv").open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {key: clean_value(value) for key, value in raw.items()}
            dataset = str(row["dataset"])
            seed = int(row["seed"])
            groups[(dataset, seed)].append(row)
            total += 1
            heavy += int(bool(row["wake_heavy_detector"]))
            reason_counts[str(row["trigger_reason"])] += 1

    fields = (
        "index",
        "uncertainty",
        "rare_probability",
        "drift_score",
        "trigger_uncertainty",
        "trigger_rare_signal",
        "trigger_drift",
        "wake_heavy_detector",
        "true_class",
        "light_prediction",
        "heavy_prediction",
        "trigger_reason",
    )
    traces: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    trace_meta: dict[str, Any] = {}
    for (dataset, seed), rows in sorted(groups.items()):
        window = best_contiguous_window(rows)
        traces[dataset][str(seed)] = compact_rows(window, fields)
        trace_meta[f"{dataset}:{seed}"] = {
            "source_events": len(rows),
            "window_events": len(window),
            "first_source_index": window[0]["index"],
            "last_source_index": window[-1]["index"],
            "heavy_events_in_window": sum(bool(row["wake_heavy_detector"]) for row in window),
        }
    summary = {
        "total_events": total,
        "heavy_events": heavy,
        "light_only_events": total - heavy,
        "overall_heavy_call_ratio": heavy / total if total else None,
        "trigger_reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "trace_windows": trace_meta,
    }
    return dict(traces), summary


def main() -> int:
    missing = [str(path.relative_to(SITE_ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise SystemExit("Missing required evidence files:\n- " + "\n- ".join(missing))

    readiness = json.loads((REPORTS_ROOT / "PUBLICATION_READINESS.json").read_text(encoding="utf-8"))
    policy_metrics = read_csv("agentic_policy_metrics.csv")
    selections = read_csv("agentic_validation_selections.csv")
    central = read_csv("central_summary_ci.csv")
    baselines = read_csv("baseline_seed_metrics.csv")
    robustness = read_csv("robustness_summary.csv")
    robustness_checks = read_csv("robustness_comparisons.csv")
    ablations = read_csv("ablation_paired_gains.csv")
    continual = read_csv("continual_comparisons.csv")
    federated = read_csv("federated_seed_metrics.csv")
    paired_gains = read_csv("paired_gain_ci.csv")
    paired_tests = read_csv("paired_statistical_tests.csv")
    traces, trace_summary = build_event_traces()

    roles = readiness["dataset_roles"]
    datasets = []
    for dataset in sorted(DATASET_LABELS):
        role = "external_confirmation" if dataset in roles["external_confirmation"] else "corrective_development"
        metadata_path = SOURCE_ROOT / "processed" / dataset / "split_metadata.json"
        split_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        datasets.append(
            {
                "id": dataset,
                "label": DATASET_LABELS[dataset],
                "role": role,
                "split_metadata": split_metadata,
            }
        )

    central_fields = [
        "dataset", "model", "seeds", "accuracy_mean", "accuracy_ci95_half",
        "f1_macro_mean", "f1_macro_ci95_half", "rare_f1_macro_mean",
        "rare_f1_macro_ci95_half", "ece_15_mean", "ece_15_ci95_half",
        "test_samples_per_second_mean", "test_samples_per_second_ci95_half",
        "parameters_mean", "peak_rss_mb_mean", "test_access_count_mean",
    ]
    policy_fields = [
        "dataset", "seed", "policy", "events", "heavy_calls", "heavy_call_ratio",
        "projected_inference_seconds_from_measured_runs", "projected_events_per_second",
        "energy_evidence", "test_access_count", "uncertainty_threshold",
        "rare_probability_threshold", "drift_threshold", "accuracy", "f1_macro",
        "rare_f1_macro", "ece_15", "balanced_accuracy", "mcc",
    ]
    selection_fields = [
        "dataset", "seed", "test_access_count", "uncertainty_threshold",
        "rare_probability_threshold", "drift_threshold", "validation_f1_macro",
        "validation_rare_f1_macro", "validation_composite",
        "validation_heavy_call_ratio", "validation_selection_objective",
        "threshold_grid_source",
    ]

    source_manifest = []
    for path in REQUIRED_FILES:
        source_manifest.append(
            {
                "path": str(path.relative_to(SITE_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "RareGuard-XIDS Research Simulation Console",
        "evidence_scope": "Final publication_v2 artifacts; bounded deterministic range samples",
        "simulation_mode": "Artifact-backed replay with optional threshold what-if on recorded scores",
        "readiness": readiness,
        "datasets": datasets,
        "seeds": sorted({int(row["seed"]) for row in policy_metrics}),
        "policy_metrics": compact_rows(policy_metrics, policy_fields),
        "validation_selections": compact_rows(selections, selection_fields),
        "central_summary": compact_rows(central, central_fields),
        "baseline_summary": aggregate_baselines(baselines),
        "robustness_summary": robustness,
        "robustness_comparisons": robustness_checks,
        "ablation_paired_gains": ablations,
        "continual_comparisons": continual,
        "federated_summary": aggregate_federated(federated),
        "paired_gain_ci": paired_gains,
        "paired_statistical_tests": paired_tests,
        "event_traces": traces,
        "event_trace_summary": trace_summary,
        "source_manifest": source_manifest,
        "disclosures": [
            "Readiness is NOT_READY because seven scientific gates remain unsupported.",
            "The replay uses recorded deterministic sample events; it is not live network inference.",
            "Inference time, throughput, and heavy-call ratio are measured or projected from measured runs; energy in joules was not measured.",
            "The experiment covers bounded deterministic range samples, not every source row.",
            "Threshold what-if results are local replay diagnostics and are not publication benchmark rows.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(SITE_ROOT)} ({OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"Recorded events verified: {trace_summary['total_events']:,}")
    print(f"Replay windows: {len(trace_summary['trace_windows'])} dataset/seed combinations")
    print(f"Source files hashed: {len(source_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
