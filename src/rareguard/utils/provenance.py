from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "torch", "matplotlib", "psutil", "PyYAML")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def package_versions() -> dict[str, str]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def inventory_files(project_root: Path, artifact_root: Path) -> list[dict[str, object]]:
    candidates: list[Path] = []
    for folder in ("results", "reports", "figures", "models"):
        directory = artifact_root / folder
        if directory.exists():
            candidates.extend(path for path in directory.rglob("*") if path.is_file() and not path.name.startswith("._"))
    processed = artifact_root / "processed"
    if processed.exists():
        for name in ("acquisition_metadata.json", "split_metadata.json", "splits.npz", "train_fitted_transformer.joblib"):
            candidates.extend(processed.glob(f"*/{name}"))
    config_dir = project_root / "configs"
    candidates.extend(path for path in config_dir.glob("*.yaml") if path.is_file() and not path.name.startswith("._"))
    rows = []
    for path in sorted(set(candidates)):
        if path.name == "PROVENANCE_MANIFEST.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(project_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_provenance_manifest(project_root: Path, artifact_root: Path) -> Path:
    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    dirty = git_output(project_root, "status", "--porcelain")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "artifact_root": str(artifact_root),
        "git": {
            "commit": git_output(project_root, "rev-parse", "HEAD"),
            "branch": git_output(project_root, "branch", "--show-current"),
            "dirty": dirty not in {"", "unavailable"},
            "dirty_paths": dirty.splitlines() if dirty not in {"", "unavailable"} else [],
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_mps_available": bool(torch.backends.mps.is_available()),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "packages": package_versions(),
        },
        "artifacts": inventory_files(project_root, artifact_root),
    }
    destination = artifact_root / "reports" / "PROVENANCE_MANIFEST.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination
