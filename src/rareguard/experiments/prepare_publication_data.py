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
    parser.add_argument("--force", action="store_true")
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
        if args.resume and not args.force and split_path.exists() and metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing_metadata.get("ready"):
                logger.info("Skipping completed publication split %s", key)
                summaries.append({"dataset": key, **existing_metadata})
                continue
            logger.info("Rebuilding rejected publication split %s", key)
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
            bool(dataset.get("label_from_source_filename", False)),
        )
        split = prepare_train_only_splits(
            raw_path,
            label,
            destination,
            int(cfg["project"]["seeds"][0]),
            float(cfg["data"]["test_size"]),
            float(cfg["data"]["validation_size"]),
            int(cfg["data"].get("duplicate_decimals", 6)),
        )
        summaries.append({"dataset": key, **acquisition, **split})
        logger.info("%s rows=%d features=%d classes=%d ready=%s", key, split["rows"], split["features"], split["classes"], split["ready"])
    summary_dir = root / "results" / "data_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        (summary_dir / f"{summary['dataset']}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    merged = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(summary_dir.glob("*.json"))
        if not path.name.startswith("._")
    ]
    (root / "results" / "publication_data_summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
