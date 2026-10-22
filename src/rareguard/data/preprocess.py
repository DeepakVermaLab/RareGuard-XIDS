from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from rareguard.config import load_config, load_yaml


@dataclass
class PreprocessResult:
    dataset_key: str
    dataset_folder: str
    rows: int
    features: int
    classes: int
    label_column: str
    output_path: str
    label_mapping_path: str


def list_csv_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in dataset_dir.rglob("*.csv")
        if p.is_file() and not p.name.startswith("._") and ".ipynb_checkpoints" not in p.parts
    )


def pick_label_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lowered = {str(c).lower(): c for c in frame.columns}
    for generic in ["label", "label_full", "label1", "attack", "class", "target", "attack_label", "attack_type"]:
        if generic in lowered:
            return str(lowered[generic])
    raise ValueError(f"No label column found. Columns include: {list(frame.columns)[:30]}")


def read_dataset_sample(csv_files: list[Path], sample_rows: int) -> pd.DataFrame:
    if not csv_files:
        raise FileNotFoundError("No CSV files found for dataset.")
    parts: list[pd.DataFrame] = []
    per_file = max(1, sample_rows // max(1, len(csv_files)))
    for path in csv_files:
        try:
            part = pd.read_csv(path, nrows=per_file, low_memory=False)
        except UnicodeDecodeError:
            part = pd.read_csv(path, nrows=per_file, low_memory=False, encoding="latin-1")
        if not part.empty:
            part["__source_file__"] = path.name
            parts.append(part)
        if sum(len(p) for p in parts) >= sample_rows:
            break
    if not parts:
        raise ValueError("CSV files were readable but empty.")
    return pd.concat(parts, ignore_index=True).head(sample_rows)


def clean_columns(frame: pd.DataFrame, max_columns: int) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().replace(" ", "_") for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    if frame.shape[1] > max_columns:
        keep = list(frame.columns[:max_columns])
        label_like = [c for c in frame.columns if c.lower() in {"label", "attack", "class", "attack_label", "attack_type"}]
        for col in label_like:
            if col not in keep:
                keep[-1] = col
        frame = frame[keep]
    return frame


def label_like_columns(columns: list[str], target: str) -> list[str]:
    label_like = {
        "label",
        "label_full",
        "label1",
        "label2",
        "label3",
        "label4",
        "attack",
        "class",
        "target",
        "attack_label",
        "attack_type",
    }
    target_lower = target.lower()
    drops: list[str] = []
    for col in columns:
        lowered = str(col).strip().replace(" ", "_").lower()
        if lowered != target_lower and lowered in label_like:
            drops.append(col)
    return drops


def build_feature_matrix(frame: pd.DataFrame, label_column: str) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    y_raw = frame[label_column].astype(str).fillna("unknown").str.strip()
    invalid = y_raw.str.lower().isin({"", "nan", "none", label_column.lower()})
    if invalid.any():
        frame = frame.loc[~invalid].copy()
        y_raw = y_raw.loc[~invalid]
    if frame.empty:
        raise ValueError(f"No valid rows remain after cleaning label column {label_column}.")
    leakage_columns = label_like_columns(list(frame.columns), label_column)
    x = frame.drop(columns=[label_column, *leakage_columns], errors="ignore")
    x = x.drop(columns=[c for c in x.columns if c.startswith("__")], errors="ignore")

    for col in x.columns:
        if x[col].dtype == object:
            x[col] = x[col].astype(str).replace({"nan": np.nan, "None": np.nan})

    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_cols),
        ],
        remainder="drop",
    )
    x_matrix = transformer.fit_transform(x)
    if hasattr(x_matrix, "toarray"):
        x_matrix = x_matrix.toarray()
    x_matrix = np.asarray(x_matrix, dtype=np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype(np.int64)
    label_mapping = {str(label): int(idx) for idx, label in enumerate(label_encoder.classes_)}
    feature_names = [f"f_{i}" for i in range(x_matrix.shape[1])]
    return x_matrix, y, feature_names, label_mapping


def preprocess_one(dataset_key: str, config_path: str = "configs/default.yaml", datasets_path: str = "configs/datasets.yaml") -> PreprocessResult:
    cfg = load_config(config_path)
    ds_cfg = load_yaml(datasets_path)["datasets"][dataset_key]
    dataset_dir = cfg.dataset_root / ds_cfg["folder"]
    frame = read_dataset_sample(list_csv_files(dataset_dir), int(cfg.raw["data"]["sample_rows_per_dataset"]))
    frame = clean_columns(frame, int(cfg.raw["data"]["max_columns"]))
    label_column = pick_label_column(frame, [c.strip().replace(" ", "_") for c in ds_cfg.get("label_candidates", [])])
    x_matrix, y, feature_names, label_mapping = build_feature_matrix(frame, label_column)

    out_dir = cfg.artifact_root / "processed" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "features.npz"
    mapping_path = out_dir / "label_mapping.json"
    np.savez_compressed(output_path, X=x_matrix, y=y, feature_names=np.asarray(feature_names, dtype=object))
    mapping_path.write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    result = PreprocessResult(
        dataset_key=dataset_key,
        dataset_folder=ds_cfg["folder"],
        rows=int(x_matrix.shape[0]),
        features=int(x_matrix.shape[1]),
        classes=len(label_mapping),
        label_column=label_column,
        output_path=str(output_path),
        label_mapping_path=str(mapping_path),
    )
    (out_dir / "preprocess_summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def preprocess_all(config_path: str = "configs/default.yaml", datasets_path: str = "configs/datasets.yaml") -> list[PreprocessResult]:
    keys = list(load_yaml(datasets_path)["datasets"].keys())
    results = [preprocess_one(key, config_path, datasets_path) for key in keys]
    summary = pd.DataFrame([asdict(r) for r in results])
    out = load_config(config_path).artifact_root / "results" / "preprocess_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    return results
