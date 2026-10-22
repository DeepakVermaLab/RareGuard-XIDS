from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from rareguard.config import load_config, load_yaml
from rareguard.data.splits import load_npz, stratified_split
from rareguard.models.train_torch import train_torch_model
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    logger = configure_logging(cfg.artifact_root / "logs" / "run_embedding.log")
    keys = list(load_yaml(args.datasets)["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    rows = []
    for key in keys:
        x, y = load_npz(cfg.artifact_root / "processed" / key / "features.npz")
        split = stratified_split(x, y, cfg.raw["data"]["test_size"], cfg.raw["data"]["validation_size"], cfg.seed)
        for model_name in ["FalconID", "RareGuard"]:
            result = train_torch_model(
                key,
                model_name,
                split.x_train,
                split.y_train,
                split.x_val,
                split.y_val,
                split.x_test,
                split.y_test,
                cfg.artifact_root / "models",
                latent_dim=int(cfg.raw["embedding"]["latent_dim"]),
                hidden_dim=int(cfg.raw["embedding"]["hidden_dim"]),
                dropout=float(cfg.raw["embedding"]["dropout"]),
                epochs=int(cfg.raw["training"]["epochs"]),
                batch_size=int(cfg.raw["training"]["batch_size"]),
                lr=float(cfg.raw["training"]["learning_rate"]),
                focal_gamma=float(cfg.raw["embedding"]["focal_gamma"]),
                contrastive_temperature=float(cfg.raw["embedding"]["contrastive_temperature"]),
                device=args.device or cfg.device,
            )
            rows.append(asdict(result))
            logger.info("%s %s f1_macro=%.4f", key, model_name, result.f1_macro)
    out = cfg.artifact_root / "results" / "embedding_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

