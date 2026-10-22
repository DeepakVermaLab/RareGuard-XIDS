from __future__ import annotations

import numpy as np
import torch

from rareguard.data.publication import parquet_compatible
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.audit_publication import full_beats_ablations, holm_adjust, paired_statistical_tests
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
