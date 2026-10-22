from __future__ import annotations

import numpy as np


def dirichlet_non_iid_partition(y: np.ndarray, n_clients: int, alpha: float = 0.5, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for client_id, part in enumerate(np.split(idx, cuts)):
            client_indices[client_id].extend(part.tolist())
    return [np.asarray(sorted(v), dtype=int) for v in client_indices]

