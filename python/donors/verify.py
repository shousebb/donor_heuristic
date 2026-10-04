"""Checks that a donor set covers every parent within tau (integer units)."""

import numpy as np
import z3

from .distances import radius


def covers(d: np.ndarray, selected: list[int], tau: int, k: int) -> bool:
    """At most k distinct donors, and every parent has one within tau."""
    return len(set(selected)) <= k and radius(d, selected) <= tau


def verify(d: np.ndarray, selected: list[int], tau: int, k: int) -> bool:
    """The same check as `covers`, proved independently by Z3."""
    t = z3.Int("tau")
    s = z3.Solver()
    s.add(t == tau)
    s.add(len(set(selected)) <= k)
    for j in range(len(d)):
        s.add(z3.Or([int(d[i, j]) <= t for i in selected]))
    return s.check() == z3.sat
