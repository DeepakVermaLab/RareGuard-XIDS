from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier


@dataclass
class MetricRow:
    dataset: str
    model: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float


def compute_metrics(dataset: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> MetricRow:
    return MetricRow(
        dataset=dataset,
        model=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )


def build_sklearn_baselines(seed: int = 42) -> dict[str, object]:
    return {
        "LogReg": LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=seed, early_stopping=True),
    }


def train_sklearn_baselines(
    dataset: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[MetricRow] = []
    for name, model in build_sklearn_baselines(seed).items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(compute_metrics(dataset, name, y_test, pred))
        joblib.dump(model, out / f"{dataset}_{name}.joblib")
    frame = pd.DataFrame([asdict(r) for r in rows])
    frame.to_csv(out / f"{dataset}_baseline_metrics.csv", index=False)
    return frame

