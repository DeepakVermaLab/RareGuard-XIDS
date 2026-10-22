from __future__ import annotations

import numpy as np
import torch

from rareguard.evaluation.metrics import classification_metrics
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
