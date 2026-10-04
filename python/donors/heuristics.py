"""Rust heuristics (donors._core): multi-start greedy + swap construction, then ALNS.

Both take the n x n matrix of integer distances from distances.to_units and give
an upper bound only.
"""

import math

import numpy as np

from . import _core
from .distances import to_bytes
from .solution import Solution


def solve_heuristic(d: np.ndarray, k: int, seeds: int = 10, seed: int = 0) -> Solution:
    """Rust multi-start greedy + swap descent selecting k donors.

    Seed 0 is the plain greedy; the others start from distinct random first
    donors. Ties on tau are broken by fewer parents at tau.
    """
    result = _core.construct(to_bytes(d), len(d), k, seeds, seed)
    donors = result["donors"]
    return Solution(
        donors=donors,
        tau=result["tau"],
        count=result["count"],
        lower=None,
        seconds=result["wall"],
        status=f"best of {seeds} seeds",
        sets=[donors],
        counts=[result["count"]],
    )


def solve_alns(
    d: np.ndarray,
    start: list[int],
    max_iters: int = 1_000_000,
    patience: int = 100_000,
    time_limit: float = math.inf,
    runs: int = 4,
    seed: int = 0,
    max_sets: int = 1,
) -> Solution:
    """Rust ALNS from `start` (k = len(start)): `runs` parallel searches.

    Each run stops after `max_iters` iterations, `patience` iterations without
    a new best (0 = off), or `time_limit` seconds, whichever comes first.
    Destroys a few donors per iteration (random, nearest the bottleneck parent,
    or related), repairs greedily with swap descent, and accepts by simulated
    annealing. Each run keeps its best `max_sets` distinct donor sets at its
    lowest tau; `sets` holds the best `max_sets` at the overall lowest tau.
    """
    result = _core.optimize(
        to_bytes(d),
        len(d),
        [int(i) for i in start],
        max_iters=max_iters,
        patience=patience,
        time_limit_s=time_limit,
        seed=seed,
        runs=runs,
        max_sets=max_sets,
    )
    return Solution(
        donors=result["donors"],
        tau=result["tau"],
        count=result["count"],
        lower=None,
        seconds=result["wall"],
        status=f"run iterations {result['run_iters']}, run taus in units {result['run_taus']}",
        sets=result["sets"],
        counts=result["counts"],
    )
