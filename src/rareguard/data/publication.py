from __future__ import annotations

import hashlib
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

from rareguard.data.preprocess import label_like_columns


IDENTIFIER_TOKENS = (
    "flow_id",
    "src_ip",
    "dst_ip",
    "ip.src",
    "ip.dst",
    "timestamp",
    "frame.time",
)


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def _run(args: list[str]) -> bytes:
    completed = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout


def list_remote_csvs(remote: str, folder: str, root_id: str, csv_root: str = "") -> list[RemoteFile]:
    listing_root = "/".join(part.strip("/") for part in [folder, csv_root] if part)
    output = _run(
        [
            "rclone",
            "lsf",
            f"{remote}{listing_root}",
            "--drive-root-folder-id",
            root_id,
            "--recursive",
            "--files-only",
            "--include",
            "*.csv",
            "--format",
            "sp",
        ]
    ).decode("utf-8", errors="replace")
    files: list[RemoteFile] = []
    for line in output.splitlines():
        if ";" not in line:
            continue
        size, relative = line.split(";", 1)
        if Path(relative).name.startswith("._"):
            continue
        files.append(RemoteFile(f"{listing_root}/{relative}", int(size)))
    return files


def _cat_range(remote: str, path: str, root_id: str, offset: int, count: int) -> bytes:
    return _run(
        [
            "rclone",
            "cat",
            f"{remote}{path}",
            "--drive-root-folder-id",
            root_id,
            "--offset",
            str(offset),
            "--count",
            str(count),
        ]
    )


def _header(remote: str, path: str, root_id: str) -> bytes:
    chunk = _cat_range(remote, path, root_id, 0, 262144)
    return chunk.splitlines()[0]


def parquet_compatible(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns so Arrow never guesses conflicting scalar types."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        result[column] = result[column].astype("string")
    return result


def source_filename_labels(paths: pd.Series) -> pd.Series:
    return paths.astype(str).map(lambda value: Path(value).stem.strip())


def sample_remote_csv(
    remote: str,
    source: RemoteFile,
    root_id: str,
    window_bytes: int,
    windows: int,
) -> pd.DataFrame:
    header = _header(remote, source.path, root_id)
    max_offset = max(0, source.size - window_bytes)
    offsets = sorted({int(value) for value in np.linspace(0, max_offset, max(1, windows))})
    frames: list[pd.DataFrame] = []
    for offset in offsets:
        payload = _cat_range(remote, source.path, root_id, offset, min(window_bytes, source.size - offset))
        lines = payload.splitlines()
        if offset > 0 and lines:
            lines = lines[1:]
        if offset + window_bytes < source.size and lines:
            lines = lines[:-1]
        if not lines:
            continue
        raw = header + b"\n" + b"\n".join(lines)
        try:
            frame = pd.read_csv(io.BytesIO(raw), low_memory=False, on_bad_lines="skip")
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(raw), low_memory=False, on_bad_lines="skip", encoding="latin-1")
        if frame.empty:
            continue
        frame["__source_file__"] = source.path
        frame["__source_offset__"] = offset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_remote_sample(
    remote: str,
    folder: str,
    root_id: str,
    label_candidates: list[str],
    output_dir: Path,
    window_bytes: int,
    windows: int,
    max_rows_per_class: int,
    max_total_rows: int,
    seed: int,
    csv_root: str = "",
    workers: int = 4,
    label_from_source_filename: bool = False,
) -> tuple[Path, str, dict[str, object]]:
    files = list_remote_csvs(remote, folder, root_id, csv_root)
    if not files:
        raise FileNotFoundError(f"No remote CSV files found under {folder}")
    part_dir = output_dir / "range_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    def load_or_sample(item: RemoteFile) -> pd.DataFrame:
        cache_key = hashlib.sha1(
            f"{item.path}|{item.size}|{window_bytes}|{windows}".encode("utf-8")
        ).hexdigest()
        cache_path = part_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        sampled_frame = sample_remote_csv(remote, item, root_id, window_bytes, windows)
        sampled_frame = parquet_compatible(sampled_frame)
        sampled_frame.to_parquet(cache_path, index=False)
        return sampled_frame

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = {executor.submit(load_or_sample, item): item for item in files}
        frames = []
        for future in as_completed(pending):
            item = pending[future]
            frame = future.result()
            frames.append(frame)
            print(f"Sampled {item.path}: {len(frame):,} rows")
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"No CSV rows could be sampled from {folder}")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame.columns = [str(col).strip().replace(" ", "_") for col in frame.columns]
    normalized_candidates = [candidate.strip().replace(" ", "_") for candidate in label_candidates]
    if label_from_source_filename:
        if "__source_file__" not in frame.columns:
            raise ValueError("Source-file label derivation requested without source provenance.")
        label = "__derived_source_label__"
        frame[label] = source_filename_labels(frame["__source_file__"])
    else:
        label = next((candidate for candidate in normalized_candidates if candidate in frame.columns), None)
    if label is None:
        lowered = {str(col).lower(): col for col in frame.columns}
        label = next((lowered[name] for name in ["label", "label4", "attack_label", "attack_type"] if name in lowered), None)
    if label is None:
        raise ValueError(f"No label column found for {folder}")
    frame[label] = frame[label].astype(str).str.strip()
    frame = frame.loc[~frame[label].str.lower().isin({"", "nan", "none", label.lower()})].copy()
    payload_columns = [col for col in frame.columns if not col.startswith("__") or col == label]
    frame = frame.drop_duplicates(subset=payload_columns).reset_index(drop=True)
    sampled = []
    for _, group in frame.groupby(label, dropna=False):
        sampled.append(group.sample(min(len(group), max_rows_per_class), random_state=seed))
    frame = pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(frame) > max_total_rows:
        selected, _ = train_test_split(
            np.arange(len(frame)),
            train_size=max_total_rows,
            random_state=seed,
            stratify=frame[label].astype(str),
        )
        frame = frame.iloc[selected].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "raw_sample.parquet"
    frame = parquet_compatible(frame)
    frame.to_parquet(path, index=False)
    metadata = {
        "folder": folder,
        "csv_root": csv_root,
        "remote_files": len(files),
        "sampled_files": len(frames),
        "rows_before_class_cap": int(sum(len(item) for item in frames)),
        "rows_after_class_cap": int(len(frame)),
        "classes": int(frame[label].nunique()),
        "label_column": label,
        "label_source": "source_filename" if label_from_source_filename else "csv_column",
        "window_bytes": window_bytes,
        "windows_per_file": windows,
        "parallel_workers": workers,
        "max_rows_per_class": max_rows_per_class,
        "max_total_rows": max_total_rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence": "remote_range_sample_from_canonical_drive_dataset",
    }
    (output_dir / "acquisition_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path, str(label), metadata


def _numeric_frame(frame: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    drops = {label_column, *label_like_columns(list(frame.columns), label_column)}
    drops.update(col for col in frame.columns if col.startswith("__"))
    drops.update(col for col in frame.columns if any(token in col.lower() for token in IDENTIFIER_TOKENS))
    numeric: dict[str, pd.Series] = {}
    rejected: list[str] = []
    for col in frame.columns:
        if col in drops:
            continue
        converted = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if float(converted.notna().mean()) >= 0.90:
            numeric[col] = converted
        else:
            rejected.append(col)
    if not numeric:
        raise ValueError("No stable numeric features remain after leakage-safe filtering.")
    return pd.DataFrame(numeric, index=frame.index), rejected


def prepare_train_only_splits(
    raw_path: Path,
    label_column: str,
    output_dir: Path,
    seed: int,
    test_size: float,
    validation_size: float,
    duplicate_decimals: int = 6,
) -> dict[str, object]:
    frame = pd.read_parquet(raw_path)
    payload_columns = [col for col in frame.columns if not col.startswith("__") or col == label_column]
    rows_before_exact_dedup = len(frame)
    frame = frame.drop_duplicates(subset=payload_columns).reset_index(drop=True)
    exact_duplicate_rows_removed = rows_before_exact_dedup - len(frame)
    labels = frame[label_column].astype(str).str.strip()
    counts = labels.value_counts()
    supported = counts[counts >= 5].index
    frame = frame.loc[labels.isin(supported)].reset_index(drop=True)
    labels = frame[label_column].astype(str).str.strip()
    numeric, rejected = _numeric_frame(frame, label_column)
    rounded_hash = pd.util.hash_pandas_object(numeric.round(duplicate_decimals), index=False).to_numpy()
    hash_labels = pd.DataFrame({"hash": rounded_hash, "label": labels.to_numpy()})
    conflicting_hashes = set(hash_labels.groupby("hash")["label"].nunique().loc[lambda values: values > 1].index)
    conflicting_mask = np.isin(rounded_hash, list(conflicting_hashes))
    duplicate_mask = pd.Series(rounded_hash).duplicated(keep="first").to_numpy()
    keep = ~conflicting_mask & ~duplicate_mask
    conflicting_feature_rows_removed = int(conflicting_mask.sum())
    feature_duplicate_rows_removed = int((duplicate_mask & ~conflicting_mask).sum())
    frame = frame.loc[keep].reset_index(drop=True)
    numeric = numeric.loc[keep].reset_index(drop=True)
    labels = frame[label_column].astype(str).str.strip()
    counts = labels.value_counts()
    supported = counts[counts >= 5].index
    supported_mask = labels.isin(supported).to_numpy()
    frame = frame.loc[supported_mask].reset_index(drop=True)
    numeric = numeric.loc[supported_mask].reset_index(drop=True)
    labels = frame[label_column].astype(str).str.strip()
    if labels.nunique() < 2:
        raise ValueError("At least two supported classes are required after feature deduplication.")
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels).astype(np.int64)
    indices = np.arange(len(frame))
    train_val, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    val_relative = validation_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed, stratify=y[train_val])
    transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler(quantile_range=(5.0, 95.0)))])
    x_train = transformer.fit_transform(numeric.iloc[train]).astype(np.float32)
    x_val = transformer.transform(numeric.iloc[val]).astype(np.float32)
    x_test = transformer.transform(numeric.iloc[test]).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "splits.npz",
        X_train=x_train,
        X_val=x_val,
        X_test=x_test,
        y_train=y[train],
        y_val=y[val],
        y_test=y[test],
        train_indices=train,
        val_indices=val,
        test_indices=test,
        feature_names=np.asarray(list(numeric.columns), dtype=object),
    )
    joblib.dump(transformer, output_dir / "train_fitted_transformer.joblib")
    row_hash = pd.util.hash_pandas_object(numeric.round(duplicate_decimals), index=False).to_numpy()
    overlaps = {
        "train_val": int(len(set(row_hash[train]) & set(row_hash[val]))),
        "train_test": int(len(set(row_hash[train]) & set(row_hash[test]))),
        "val_test": int(len(set(row_hash[val]) & set(row_hash[test]))),
    }
    metadata = {
        "seed": seed,
        "rows": int(len(frame)),
        "features": int(numeric.shape[1]),
        "classes": int(len(encoder.classes_)),
        "class_mapping": {str(label): int(i) for i, label in enumerate(encoder.classes_)},
        "split_rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "split_class_counts": {
            "train": np.bincount(y[train], minlength=len(encoder.classes_)).tolist(),
            "validation": np.bincount(y[val], minlength=len(encoder.classes_)).tolist(),
            "test": np.bincount(y[test], minlength=len(encoder.classes_)).tolist(),
        },
        "rejected_non_numeric_columns": rejected,
        "duplicate_decimals": duplicate_decimals,
        "exact_duplicate_rows_removed": int(exact_duplicate_rows_removed),
        "feature_duplicate_rows_removed": feature_duplicate_rows_removed,
        "conflicting_feature_rows_removed": conflicting_feature_rows_removed,
        "duplicate_hash_overlap": overlaps,
        "transform_fit_scope": "training_only",
        "test_locked": True,
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "ready": all(value == 0 for value in overlaps.values()),
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
