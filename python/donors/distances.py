"""Genetic distances: synthetic data and the integer units the solvers use.

Distances are scaled by `scale` (the [solver] scale setting, default SCALE)
and rounded, so tau is an integer and every answer is exact to 1 / scale.
"""

import numpy as np

SCALE = 100  # default: distances are solved in integer units of 1 / SCALE


def get_distances(
    n: int, mu: float, sigma: float, bounds: tuple[float, float], seed: int = 42
) -> np.ndarray:
    """Genetic distances 1 - similarity.

    Similarity ~ N(mu, sigma) clipped to bounds, symmetric, 1 on the diagonal,
    so distances are symmetric with a zero diagonal.
    """
    rng = np.random.default_rng(seed)
    upper = np.triu(np.clip(rng.normal(mu, sigma, (n, n)), *bounds), k=1)
    similarity = upper + upper.T + np.eye(n)
    return 1 - similarity


def to_units(d: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Integer distances in units of 1 / scale; at most scale + 1 distinct values."""
    return np.rint(d * scale).astype(int)


def to_bytes(d: np.ndarray) -> bytes:
    """Integer distances as the little-endian u16 buffer the Rust kernel reads."""
    assert d.max() <= np.iinfo(np.uint16).max, "scaled distances must fit in u16"
    return d.astype("<u2").tobytes()


def radius(d: np.ndarray, selected: list[int]) -> int:
    """tau of a selection: the largest distance from a parent to its nearest donor."""
    return int(d[:, selected].min(axis=1).max())


def parents_at_tau(d: np.ndarray, selected: list[int]) -> int:
    """Number of parents whose nearest selected donor is exactly tau away."""
    nearest = d[:, selected].min(axis=1)
    return int((nearest == nearest.max()).sum())
