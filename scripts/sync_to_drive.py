#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rareguard.config import load_config


EXCLUDES = ["._*", ".DS_Store", "__pycache__/**", ".ipynb_checkpoints/**"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync RareGuard-XIDS artifacts to the configured Google Drive folder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default="artifacts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = Path(args.source)
    if shutil.which("rclone") is None:
        print("rclone is not installed in this runtime.")
        if str(Path.cwd()).startswith("/content/drive/"):
            print("Artifacts are already being written inside Google Drive because the project root is under /content/drive.")
            print(f"Current artifact folder: {source.resolve()}")
            print("Skipping external rclone mirror. Continue with the next notebook.")
            return 0
        print("Install rclone or run this sync script on the local Mac where rclone is configured.")
        return 2
    destination = f"{cfg.raw['paths']['rclone_remote']}{cfg.raw['paths']['drive_project_path']}/artifacts"
    cmd = ["rclone", "copy", str(source), destination, "--drive-root-folder-id", cfg.raw["paths"]["drive_folder_id"], "--progress", "--transfers", "2", "--checkers", "4", "--size-only"]
    for pattern in EXCLUDES:
        cmd.extend(["--exclude", pattern])
    if args.dry_run:
        cmd.append("--dry-run")
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
