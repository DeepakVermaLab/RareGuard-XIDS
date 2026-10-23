from __future__ import annotations

import numpy as np

from rareguard.agent.controller import RareGuardAgent
from rareguard.agent.runtime import ContinuousRareGuardRuntime, RuntimePaths
from rareguard.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.raw["project"]["name"] == "RareGuard-XIDS"
    assert cfg.dataset_root.name == "Security"


def test_agent_decision_shape() -> None:
    agent = RareGuardAgent(
        uncertainty_threshold=0.35,
        rare_probability_threshold=0.25,
        drift_threshold=0.15,
        energy_low_budget_threshold=0.25,
    )
    decisions = agent.decide(
        probabilities=np.array([[0.8, 0.2], [0.45, 0.55]]),
        rare_class_ids={1},
        drift_scores=np.array([0.01, 0.3]),
        energy_budgets=np.array([0.9, 0.8]),
    )
    assert len(decisions) == 2
    assert decisions[1].action == "wake_heavy_detector"


def test_runtime_ignores_appledouble_inbox_files(tmp_path) -> None:
    runtime = object.__new__(ContinuousRareGuardRuntime)
    runtime.paths = RuntimePaths(
        inbox=tmp_path / "inbox",
        decisions=tmp_path / "decisions.jsonl",
        state=tmp_path / "state.json",
        heartbeat=tmp_path / "heartbeat.json",
    )
    runtime.device = "cpu"
    runtime.paths.inbox.mkdir()
    (runtime.paths.inbox / "._shadow.csv").write_bytes(b"\x00\x05AppleDouble")
    assert runtime.run_once()["status"] == "healthy"
