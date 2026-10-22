from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            score += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(score)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    rare_class_ids: set[int] | None = None,
) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    labels = np.arange(probabilities.shape[1])
    per_class_f1 = f1_score(y_true, prediction, labels=labels, average=None, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision_macro": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "cohen_kappa": float(cohen_kappa_score(y_true, prediction)),
        "ece_15": expected_calibration_error(y_true, probabilities, bins=15),
    }
    rare = sorted(rare_class_ids or set())
    result["rare_class_count"] = float(len(rare))
    result["rare_f1_macro"] = float(per_class_f1[rare].mean()) if rare else float("nan")
    try:
        binary = label_binarize(y_true, classes=labels)
        if probabilities.shape[1] == 2 and binary.shape[1] == 1:
            binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
        result["roc_auc_macro_ovr"] = float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
        result["pr_auc_macro"] = float(average_precision_score(binary, probabilities, average="macro"))
    except ValueError:
        result["roc_auc_macro_ovr"] = float("nan")
        result["pr_auc_macro"] = float("nan")
    return result
