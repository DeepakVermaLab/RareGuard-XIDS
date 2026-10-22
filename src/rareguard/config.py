from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DATASET_ROOT_FALLBACKS = [
    Path("/Volumes/LocalDrive1/Datasets/Security"),
    Path("/content/drive/MyDrive/projects/Dataset/Security"),
    Path("/content/drive/MyDrive/Dataset/Security"),
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def artifact_root(self) -> Path:
        return Path(self.raw["paths"]["artifact_root"])

    @property
    def dataset_root(self) -> Path:
        return Path(self.raw["paths"]["dataset_root"])

    @property
    def seed(self) -> int:
        return int(self.raw.get("project", {}).get("seed", 42))

    @property
    def device(self) -> str:
        return str(self.raw.get("project", {}).get("device", "cpu"))


def load_config(path: str | Path = "configs/default.yaml", override_path: str | Path | None = None) -> RuntimeConfig:
    path = Path(path)
    config = load_yaml(path)
    if override_path:
        config = deep_update(config, load_yaml(override_path))
    dataset_root = Path(config["paths"]["dataset_root"])
    if not dataset_root.exists():
        for fallback in DATASET_ROOT_FALLBACKS:
            if fallback.exists():
                config["paths"]["dataset_root"] = str(fallback)
                break
    return RuntimeConfig(raw=config, path=path)
