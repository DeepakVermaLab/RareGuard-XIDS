from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from rareguard.agent.controller import RareGuardAgent, decisions_to_frame
from rareguard.agent.energy import estimate_energy, random_energy_budgets
from rareguard.config import load_config, load_yaml
from rareguard.data.splits import load_npz
from rareguard.features.rare_profile import compute_rare_profile
from rareguard.models.torch_models import RareGuardNet
from rareguard.utils.logging import configure_logging
from rareguard.utils.seed import set_seed


def pseudo_probabilities(y: np.ndarray, classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(classes), size=len(y))
    probs[np.arange(len(y)), y] += 0.8
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def model_probabilities(dataset: str, x: np.ndarray, y: np.ndarray, cfg, device: str) -> tuple[np.ndarray, str]:
    model_path = cfg.artifact_root / "models" / f"{dataset}_RareGuard.pt"
    classes = int(np.max(y)) + 1
    if not model_path.exists():
        return pseudo_probabilities(y, classes, cfg.seed), "pseudo_fallback_model_missing"
    model = RareGuardNet(
        x.shape[1],
        classes,
        int(cfg.raw["embedding"]["latent_dim"]),
        int(cfg.raw["embedding"]["hidden_dim"]),
        float(cfg.raw["embedding"]["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            xb = torch.tensor(x[start : start + 2048], dtype=torch.float32, device=device)
            logits, _ = model(xb)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks), "trained_rareguard_model"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    logger = configure_logging(cfg.artifact_root / "logs" / "run_agentic.log")
    keys = list(load_yaml(args.datasets)["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    reports = []
    for key in keys:
        x, y = load_npz(cfg.artifact_root / "processed" / key / "features.npz")
        profiles = compute_rare_profile(y, float(cfg.raw["data"]["min_rare_class_ratio"]))
        rare_ids = {p.class_id for p in profiles if p.is_rare}
        probs, probability_source = model_probabilities(key, x, y, cfg, device)
        drift = np.clip(np.abs(np.gradient(probs.max(axis=1))), 0, 1)
        energy = random_energy_budgets(len(y), cfg.seed)
        agent = RareGuardAgent(
            float(cfg.raw["agent"]["uncertainty_threshold"]),
            float(cfg.raw["agent"]["rare_probability_threshold"]),
            float(cfg.raw["agent"]["drift_threshold"]),
            float(cfg.raw["agent"]["energy_low_budget_threshold"]),
        )
        decisions = decisions_to_frame(agent.decide(probs, rare_ids, drift, energy))
        out_decisions = cfg.artifact_root / "results" / f"{key}_agent_decisions.csv"
        decisions.to_csv(out_decisions, index=False)
        report = estimate_energy(decisions["action"])
        row = {"dataset": key, "probability_source": probability_source, **asdict(report), "decisions_path": str(out_decisions)}
        reports.append(row)
        logger.info("%s source=%s heavy_saved=%s energy_reduction=%.4f", key, probability_source, report.heavy_calls_saved, report.estimated_energy_reduction_ratio)
    out = cfg.artifact_root / "results" / "agentic_energy_summary.csv"
    pd.DataFrame(reports).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
