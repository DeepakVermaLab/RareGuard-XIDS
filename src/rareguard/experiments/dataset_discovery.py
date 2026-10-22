from __future__ import annotations

import argparse
from pathlib import Path

from rareguard.config import load_config
from rareguard.data.discovery import profile_datasets, write_profiles
from rareguard.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = configure_logging(cfg.artifact_root / "logs" / "dataset_discovery.log")
    profiles = profile_datasets(args.config, args.datasets)
    out = write_profiles(profiles, cfg.artifact_root / "results" / "dataset_profiles.csv")
    logger.info("Wrote dataset profiles to %s", out)
    for profile in profiles:
        logger.info("%s files=%s bytes=%.3f GiB labels=%s", profile.folder, profile.file_count, profile.bytes_total / 1024**3, profile.detected_label_columns)
    if profiles and not any(profile.exists for profile in profiles):
        logger.error("No configured dataset folders were found. Resolved dataset_root=%s", cfg.dataset_root)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
