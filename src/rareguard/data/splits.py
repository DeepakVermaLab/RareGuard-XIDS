from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class SplitBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
) -> SplitBundle:
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    stratify_tv = y_train_val if min(np.bincount(y_train_val)) >= 2 else None
    val_relative = validation_size / max(1e-9, 1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_relative, random_state=seed, stratify=stratify_tv
    )
    return SplitBundle(x_train, x_val, x_test, y_train, y_val, y_test)


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]

