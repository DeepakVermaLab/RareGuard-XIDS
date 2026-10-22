#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rareguard.config import load_config


def main() -> int:
    cfg = load_config("configs/default.yaml")
    for name in ["logs", "results", "processed", "features", "models", "figures", "reports"]:
        (cfg.artifact_root / name).mkdir(parents=True, exist_ok=True)

    packages = {package: bool(importlib.util.find_spec(package)) for package in ["numpy", "pandas", "sklearn", "yaml", "torch", "matplotlib"]}
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "packages": packages,
        "rclone": shutil.which("rclone"),
        "dataset_root": str(cfg.dataset_root),
        "dataset_root_exists": cfg.dataset_root.exists(),
        "artifact_root": str(cfg.artifact_root),
    }

    print("python", report["python"])
    for package, ok in packages.items():
        print(package, "OK" if ok else "MISSING")
    print("rclone", report["rclone"] or "MISSING")
    print("cwd", report["cwd"])
    print("dataset_root", report["dataset_root"])
    print("dataset_root_exists", report["dataset_root_exists"])
    print("artifact_root", report["artifact_root"])

    (cfg.artifact_root / "results" / "environment_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# RareGuard-XIDS Environment Check",
        f"checked_at_utc: {report['checked_at_utc']}",
        f"cwd: {report['cwd']}",
        f"dataset_root: {report['dataset_root']}",
        f"dataset_root_exists: {report['dataset_root_exists']}",
        f"artifact_root: {report['artifact_root']}",
        f"rclone: {report['rclone'] or 'MISSING'}",
        "",
        "## Packages",
    ]
    lines.extend(f"- {package}: {'OK' if ok else 'MISSING'}" for package, ok in packages.items())
    (cfg.artifact_root / "logs" / "environment_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
