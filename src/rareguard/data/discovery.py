from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from rareguard.config import load_config, load_yaml


@dataclass
class DatasetProfile:
    key: str
    folder: str
    path: str
    exists: bool
    manifest_status: str
    file_count: int
    csv_count: int
    parquet_count: int
    archive_count: int
    bytes_total: int
    label_candidates: list[str]
    detected_label_columns: list[str]


def iter_payload_files(dataset_dir: Path) -> Iterable[Path]:
    if not dataset_dir.exists():
        return []
    return (
        p
        for p in dataset_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != ".DS_Store"
        and ".ipynb_checkpoints" not in p.parts
    )


def discover_label_columns(files: list[Path], candidates: list[str], sample_files: int = 50) -> list[str]:
    found: set[str] = set()
    csv_files = [p for p in sorted(files) if p.suffix.lower() == ".csv"]
    normalized_candidates = {str(c).strip().replace(" ", "_").lower(): str(c).strip().replace(" ", "_") for c in candidates}
    generic = {"label", "label_full", "label1", "label2", "attack", "class", "target", "attack_label", "attack_type"}
    for path in csv_files[:sample_files]:
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        columns = {str(c).strip().replace(" ", "_"): str(c) for c in header.columns}
        lowered = {clean.lower(): original for clean, original in columns.items()}
        for clean_candidate, original_candidate in normalized_candidates.items():
            if clean_candidate in lowered:
                found.add(str(lowered[clean_candidate]))
        for name in generic:
            if name in lowered:
                found.add(str(lowered[name]))
    return sorted(found)


def profile_datasets(default_config: str = "configs/default.yaml", datasets_config: str = "configs/datasets.yaml") -> list[DatasetProfile]:
    cfg = load_config(default_config)
    ds_cfg = load_yaml(datasets_config)
    root = cfg.dataset_root
    profiles: list[DatasetProfile] = []

    for key, spec in ds_cfg["datasets"].items():
        dataset_dir = root / spec["folder"]
        files = list(iter_payload_files(dataset_dir))
        manifest = dataset_dir / "download_manifest.json"
        manifest_status = "missing"
        if manifest.exists():
            try:
                manifest_status = json.loads(manifest.read_text()).get("status", "unknown")
            except Exception:
                manifest_status = "unreadable"

        profiles.append(
            DatasetProfile(
                key=key,
                folder=spec["folder"],
                path=str(dataset_dir),
                exists=dataset_dir.exists(),
                manifest_status=manifest_status,
                file_count=len(files),
                csv_count=sum(p.suffix.lower() == ".csv" for p in files),
                parquet_count=sum(p.suffix.lower() == ".parquet" for p in files),
                archive_count=sum(p.suffix.lower() in {".zip", ".rar", ".7z"} for p in files),
                bytes_total=sum(p.stat().st_size for p in files),
                label_candidates=list(spec.get("label_candidates", [])),
                detected_label_columns=discover_label_columns(files, list(spec.get("label_candidates", []))),
            )
        )
    return profiles


def profiles_to_frame(profiles: list[DatasetProfile]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in profiles])


def write_profiles(profiles: list[DatasetProfile], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = profiles_to_frame(profiles)
    frame.to_csv(path, index=False)
    return path
