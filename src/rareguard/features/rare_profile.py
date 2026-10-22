from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RareClassProfile:
    class_id: int
    count: int
    ratio: float
    is_rare: bool


def compute_rare_profile(y: np.ndarray, min_ratio: float = 0.01) -> list[RareClassProfile]:
    values, counts = np.unique(y, return_counts=True)
    total = max(1, int(counts.sum()))
    return [
        RareClassProfile(int(v), int(c), float(c / total), bool(c / total < min_ratio))
        for v, c in zip(values, counts)
    ]


def write_rare_profile(y: np.ndarray, output_path: str | Path, min_ratio: float = 0.01) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(p) for p in compute_rare_profile(y, min_ratio)]).to_csv(path, index=False)
    return path

