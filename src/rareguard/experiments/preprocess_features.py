from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from rareguard.config import load_config, load_yaml
from rareguard.data.preprocess import preprocess_all, preprocess_one
from rareguard.data.splits import load_npz
from rareguard.features.rare_profile import write_rare_profile
from rareguard.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = configure_logging(cfg.artifact_root / "logs" / "preprocess_features.log")
    keys = list(load_yaml(args.datasets)["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    results = [preprocess_one(key, args.config, args.datasets) for key in keys]
    for result in results:
        x, y = load_npz(result.output_path)
        profile_path = cfg.artifact_root / "results" / f"{result.dataset_key}_rare_profile.csv"
        write_rare_profile(y, profile_path, float(cfg.raw["data"]["min_rare_class_ratio"]))
        logger.info("%s rows=%s features=%s classes=%s rare_profile=%s", result.dataset_key, x.shape[0], x.shape[1], result.classes, profile_path)
    out = cfg.artifact_root / "results" / "preprocess_summary.csv"
    pd.DataFrame([asdict(r) for r in results]).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

