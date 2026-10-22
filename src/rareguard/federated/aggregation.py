from __future__ import annotations

import copy

import torch


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No client states supplied.")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    total = sum(weights)
    weights = [w / total for w in weights]
    avg = copy.deepcopy(states[0])
    for key in avg:
        avg[key] = sum(state[key].detach().cpu() * float(w) for state, w in zip(states, weights))
    return avg


def energy_aware_weights(sample_counts: list[int], energy_scores: list[float]) -> list[float]:
    raw = [max(1, c) * max(0.05, e) for c, e in zip(sample_counts, energy_scores)]
    total = sum(raw)
    return [r / total for r in raw]

