from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rareguard.config import load_yaml
from rareguard.data.publication import build_remote_sample, prepare_train_only_splits
from rareguard.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build leakage-safe publication splits from canonical Drive datasets.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    datasets = load_yaml(args.datasets)["datasets"]
    keys = list(datasets) if args.dataset == "all" else [args.dataset]
    root = Path(cfg["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "prepare_publication_data.log")
    summaries = []
    for key in keys:
        destination = root / "processed" / key
        split_path = destination / "splits.npz"
        metadata_path = destination / "split_metadata.json"
        if args.resume and split_path.exists() and metadata_path.exists():
            logger.info("Skipping completed publication split %s", key)
            summaries.append({"dataset": key, **json.loads(metadata_path.read_text(encoding="utf-8"))})
            continue
        dataset = datasets[key]
        raw_path, label, acquisition = build_remote_sample(
            cfg["paths"]["rclone_remote"],
            dataset["folder"],
            cfg["paths"]["dataset_drive_root_id"],
            dataset.get("label_candidates", []),
            destination,
            int(cfg["data"]["remote_window_bytes"]),
            int(cfg["data"]["remote_windows_per_file"]),
            int(cfg["data"]["max_rows_per_class"]),
            int(cfg["data"]["max_total_rows"]),
            int(cfg["project"]["seeds"][0]),
            str(dataset.get("remote_csv_root", "")),
            int(cfg["data"].get("remote_workers", 4)),
        )
        split = prepare_train_only_splits(
            raw_path,
            label,
            destination,
            int(cfg["project"]["seeds"][0]),
            float(cfg["data"]["test_size"]),
            float(cfg["data"]["validation_size"]),
        )
        summaries.append({"dataset": key, **acquisition, **split})
        logger.info("%s rows=%d features=%d classes=%d ready=%s", key, split["rows"], split["features"], split["classes"], split["ready"])
    summary_path = root / "results" / "publication_data_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
