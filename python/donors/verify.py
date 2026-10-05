"""Cover checks: a fast numpy check and an independent CP-SAT one (integer units)."""

import numpy as np

from .cpsat import check_tau
from .distances import radius


def covers(d: np.ndarray, selected: list[int], tau: int, k: int) -> bool:
    """At most k distinct donors, and every parent has one within tau."""
    return len(set(selected)) <= k and radius(d, selected) <= tau


def verify(
    d: np.ndarray, selected: list[int], tau: int, k: int, timeout: float = 60
) -> bool:
    """The same check as `covers`, proved independently by a CP-SAT model."""
    return (
        len(set(selected)) <= k
        and check_tau(d, k, tau, timeout, donors=selected).result == "feasible"
    )
