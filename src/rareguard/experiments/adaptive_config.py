from __future__ import annotations

import copy
import json
from pathlib import Path


SELECTED_FIELDS = ("focal_mix", "rare_sampler_power", "contrastive_weight")


def selected_hyperparameters_path(artifact_root: Path, dataset: str) -> Path:
    return artifact_root / "reports" / f"SELECTED_HYPERPARAMETERS_{dataset}.json"


def apply_dataset_hyperparameters(config: dict, artifact_root: Path, dataset: str) -> dict:
    path = selected_hyperparameters_path(artifact_root, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation-selected hyperparameters missing for {dataset}: {path}. "
            "Run run_publication_adaptive_tuning before locked-test experiments."
        )
    selected = json.loads(path.read_text(encoding="utf-8"))
    if str(selected.get("dataset")) != dataset:
        raise RuntimeError(f"Rejected hyperparameters for {dataset}: selection artifact names another dataset.")
    if int(selected.get("test_access_count", -1)) != 0:
        raise RuntimeError(f"Rejected hyperparameters for {dataset}: test data were accessed during selection.")
    missing = [field for field in SELECTED_FIELDS if field not in selected]
    if missing:
        raise KeyError(f"Selected hyperparameters for {dataset} are missing {missing}")
    result = copy.deepcopy(config)
    result["training"].update({field: float(selected[field]) for field in SELECTED_FIELDS})
    return result
