from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from rareguard.config import load_yaml
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.run_publication_central import load_splits
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


def build_models(seed: int) -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=seed,
        ),
    }


def aligned_probabilities(model, x: np.ndarray, classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    probabilities = np.zeros((len(x), classes), dtype=float)
    probabilities[:, np.asarray(model.classes_, dtype=int)] = raw
    return probabilities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run leakage-safe classical publication baselines.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    logger = configure_logging(root / "logs" / "run_publication_baselines.log")
    keys = list(load_yaml(args.datasets)["datasets"]) if args.dataset == "all" else [args.dataset]
    result_path = root / "results" / "baseline_seed_metrics.csv"
    rows = pd.read_csv(result_path).to_dict("records") if args.resume and result_path.exists() else []
    complete = {(str(row["dataset"]), int(row["seed"]), str(row["model"])) for row in rows}
    for dataset in keys:
        arrays = load_splits(root / "processed" / dataset / "splits.npz")
        classes = int(max(arrays["y_train"].max(), arrays["y_test"].max())) + 1
        counts = np.bincount(arrays["y_train"], minlength=classes)
        cutoff = max(1, int(float(config["data"]["min_rare_class_ratio"]) * len(arrays["y_train"])))
        rare_ids = {int(i) for i, count in enumerate(counts) if count <= cutoff}
        for seed in [int(value) for value in config["project"]["seeds"]]:
            set_seed(seed)
            for name, model in build_models(seed).items():
                if (dataset, seed, name) in complete:
                    continue
                logger.info("Starting %s seed=%d %s", dataset, seed, name)
                started = time.perf_counter()
                model.fit(arrays["X_train"], arrays["y_train"])
                training_seconds = time.perf_counter() - started
                test_started = time.perf_counter()
                probabilities = aligned_probabilities(model, arrays["X_test"], classes)
                test_seconds = time.perf_counter() - test_started
                model_dir = root / "models" / dataset / f"seed_{seed}"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{name}.joblib"
                joblib.dump(model, model_path)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": name,
                        "training_seconds": training_seconds,
                        "test_seconds": test_seconds,
                        "test_samples_per_second": float(len(arrays["y_test"]) / max(test_seconds, 1e-9)),
                        "model_path": str(model_path),
                        "test_access_count": 1,
                        **classification_metrics(arrays["y_test"], probabilities, rare_ids),
                    }
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                logger.info("%s seed=%d %s f1_macro=%.4f", dataset, seed, name, rows[-1]["f1_macro"])
                gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
