from __future__ import annotations

import numpy as np
import torch

from rareguard.data.publication import parquet_compatible, source_filename_labels
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.audit_publication import agent_policy_value, continual_superiority, full_beats_ablations, holm_adjust, paired_statistical_tests
from rareguard.experiments.run_publication_agentic import projected_seconds
from rareguard.experiments.run_publication_continual import class_incremental_tasks, update_memory
from rareguard.models.publication_models import NovaXIDSNet, RareGuardXIDSNet


def test_publication_models_share_output_contract() -> None:
    x = torch.randn(6, 12)
    nova = NovaXIDSNet(12, 3)
    rareguard = RareGuardXIDSNet(12, 3)
    assert nova(x).shape == (6, 3)
    logits, vectors = rareguard(x)
    assert logits.shape == (6, 3)
    assert vectors.shape == (6, 64)


def test_publication_metrics_include_rare_and_calibration() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    metrics = classification_metrics(np.asarray([0, 1, 1]), probabilities, {1})
    assert metrics["f1_macro"] == 1.0
    assert metrics["rare_f1_macro"] == 1.0
    assert 0.0 <= metrics["ece_15"] <= 1.0


def test_ablation_gate_requires_full_model_to_win_every_comparison() -> None:
    import pandas as pd

    rows = []
    for seed in (42, 73):
        rows.extend(
            [
                {"dataset": "demo", "seed": seed, "model": "RareGuard-XIDS", "f1_macro": 0.8, "rare_f1_macro": 0.7},
                {"dataset": "demo", "seed": seed, "model": "RareGuard-no-focal", "f1_macro": 0.7, "rare_f1_macro": 0.6},
            ]
        )
    passed, comparisons = full_beats_ablations(pd.DataFrame(rows))
    assert passed
    assert len(comparisons) == 2


def test_paired_statistics_and_holm_adjustment() -> None:
    import pandas as pd

    rows = []
    for seed, gain in zip((42, 73, 101, 137, 211), (0.01, 0.02, 0.03, 0.01, 0.02)):
        for model, offset in (("NOVA-XIDS", 0.0), ("RareGuard-XIDS", gain)):
            rows.append(
                {
                    "dataset": "demo",
                    "seed": seed,
                    "model": model,
                    "f1_macro": 0.7 + offset,
                    "rare_f1_macro": 0.5 + offset,
                    "balanced_accuracy": 0.65 + offset,
                    "mcc": 0.6 + offset,
                    "ece_15": 0.1 - offset,
                }
            )
    result = paired_statistical_tests(pd.DataFrame(rows))
    assert len(result) == 5
    assert result["holm_adjusted_p"].notna().all()
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_parquet_compatibility_normalizes_mixed_objects() -> None:
    import pandas as pd

    frame = pd.DataFrame({"mixed": pd.Series([1, b"2", None], dtype=object), "numeric": [1.0, 2.0, 3.0]})
    normalized = parquet_compatible(frame)
    assert str(normalized["mixed"].dtype) == "string"
    assert normalized["numeric"].dtype == frame["numeric"].dtype


def test_source_filename_labels_preserve_attack_types() -> None:
    import pandas as pd

    labels = source_filename_labels(pd.Series(["root/DoS UDP Flood.csv", "root/Benign Traffic.csv"]))
    assert labels.tolist() == ["DoS UDP Flood", "Benign Traffic"]


def test_continual_tasks_and_balanced_memory() -> None:
    labels = np.asarray([0] * 20 + [1] * 10 + [2] * 5 + [3] * 2)
    tasks = class_incremental_tasks(labels, 3)
    assert sorted(value for task in tasks for value in task) == [0, 1, 2, 3]
    memory = update_memory("RareGuard-Continual", np.arange(len(labels)), labels, 8, np.random.default_rng(42))
    counts = np.bincount(labels[memory], minlength=4)
    assert counts.tolist() == [2, 2, 2, 2]


def test_continual_superiority_requires_better_rare_f1_and_forgetting() -> None:
    import pandas as pd

    rows = []
    for method, rare_f1, forgetting in (
        ("RareGuard-FineTune", 0.4, 0.30),
        ("RareGuard-ReservoirReplay", 0.5, 0.20),
        ("RareGuard-Continual", 0.6, 0.10),
    ):
        rows.append(
            {
                "dataset": "demo",
                "seed": 42,
                "method": method,
                "rare_f1_macro": rare_f1,
                "mean_validation_forgetting": forgetting,
            }
        )
    passed, comparisons = continual_superiority(pd.DataFrame(rows))
    assert passed
    assert len(comparisons) == 4


def test_agent_timing_and_policy_gate() -> None:
    import pandas as pd

    assert projected_seconds("always_light_nova", 10, 0, 0.1, 0.5) == 1.0
    assert projected_seconds("always_heavy_rareguard", 10, 10, 0.1, 0.5) == 5.0
    assert projected_seconds("rareguard_agent", 10, 2, 0.1, 0.5) == 2.0
    rows = [
        {"dataset": "demo", "policy": "always_light_nova", "f1_macro": 0.7, "rare_f1_macro": 0.4, "heavy_call_ratio": 0.0},
        {"dataset": "demo", "policy": "always_heavy_rareguard", "f1_macro": 0.9, "rare_f1_macro": 0.8, "heavy_call_ratio": 1.0},
        {"dataset": "demo", "policy": "rareguard_agent", "f1_macro": 0.8, "rare_f1_macro": 0.6, "heavy_call_ratio": 0.3},
    ]
    passed, comparisons = agent_policy_value(pd.DataFrame(rows))
    assert passed
    assert len(comparisons) == 3
