from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rareguard.config import load_config, load_yaml
from rareguard.data.splits import load_npz, stratified_split
from rareguard.models.baselines import train_sklearn_baselines
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    logger = configure_logging(cfg.artifact_root / "logs" / "run_baselines.log")
    keys = list(load_yaml(args.datasets)["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    frames: list[pd.DataFrame] = []
    for key in keys:
        npz = cfg.artifact_root / "processed" / key / "features.npz"
        x, y = load_npz(npz)
        split = stratified_split(x, y, cfg.raw["data"]["test_size"], cfg.raw["data"]["validation_size"], cfg.seed)
        frame = train_sklearn_baselines(key, split.x_train, split.y_train, split.x_test, split.y_test, cfg.artifact_root / "models", cfg.seed)
        frames.append(frame)
        logger.info("Finished baselines for %s", key)
    out = cfg.artifact_root / "results" / "baseline_metrics.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

