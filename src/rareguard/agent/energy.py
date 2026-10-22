from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class EnergyReport:
    samples: int
    heavy_calls_baseline: int
    heavy_calls_agent: int
    heavy_calls_saved: int
    heavy_call_reduction_ratio: float
    estimated_energy_baseline: float
    estimated_energy_agent: float
    estimated_energy_reduction_ratio: float


def estimate_energy(
    actions: pd.Series,
    light_cost: float = 1.0,
    heavy_cost: float = 8.0,
) -> EnergyReport:
    samples = int(len(actions))
    heavy_agent = int(actions.isin(["wake_heavy_detector", "defer_heavy_detector_and_alert_low_energy"]).sum())
    heavy_baseline = samples
    baseline_energy = samples * heavy_cost
    agent_energy = samples * light_cost + heavy_agent * heavy_cost
    saved = heavy_baseline - heavy_agent
    return EnergyReport(
        samples=samples,
        heavy_calls_baseline=heavy_baseline,
        heavy_calls_agent=heavy_agent,
        heavy_calls_saved=saved,
        heavy_call_reduction_ratio=float(saved / max(1, heavy_baseline)),
        estimated_energy_baseline=float(baseline_energy),
        estimated_energy_agent=float(agent_energy),
        estimated_energy_reduction_ratio=float((baseline_energy - agent_energy) / max(1.0, baseline_energy)),
    )


def random_energy_budgets(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.15, 1.0, size=n)

