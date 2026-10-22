from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class AgentDecision:
    index: int
    action: str
    uncertainty: float
    rare_probability: float
    drift_score: float
    energy_budget: float
    reason: str


class RareGuardAgent:
    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        rare_probability_threshold: float = 0.25,
        drift_threshold: float = 0.15,
        energy_low_budget_threshold: float = 0.25,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.rare_probability_threshold = rare_probability_threshold
        self.drift_threshold = drift_threshold
        self.energy_low_budget_threshold = energy_low_budget_threshold

    def decide(
        self,
        probabilities: np.ndarray,
        rare_class_ids: set[int],
        drift_scores: np.ndarray,
        energy_budgets: np.ndarray,
    ) -> list[AgentDecision]:
        decisions: list[AgentDecision] = []
        rare_ids = sorted(rare_class_ids)
        for i, probs in enumerate(probabilities):
            confidence = float(np.max(probs))
            uncertainty = 1.0 - confidence
            rare_prob = float(np.max(probs[rare_ids])) if rare_ids else 0.0
            drift = float(drift_scores[i])
            energy = float(energy_budgets[i])
            reasons: list[str] = []
            if uncertainty >= self.uncertainty_threshold:
                reasons.append("low_confidence")
            if rare_prob >= self.rare_probability_threshold:
                reasons.append("rare_attack_signal")
            if drift >= self.drift_threshold:
                reasons.append("traffic_drift")

            if reasons and energy >= self.energy_low_budget_threshold:
                action = "wake_heavy_detector"
            elif reasons:
                action = "defer_heavy_detector_and_alert_low_energy"
            else:
                action = "keep_light_detector"
                reasons.append("normal_budgeted_monitoring")

            decisions.append(
                AgentDecision(
                    index=i,
                    action=action,
                    uncertainty=uncertainty,
                    rare_probability=rare_prob,
                    drift_score=drift,
                    energy_budget=energy,
                    reason=";".join(reasons),
                )
            )
        return decisions


def decisions_to_frame(decisions: list[AgentDecision]) -> pd.DataFrame:
    return pd.DataFrame([asdict(d) for d in decisions])

