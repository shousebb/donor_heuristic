"""Type stubs for the Rust extension (src/lib.rs): donor-selection heuristics.

Distances are passed as the n x n matrix of scaled integer distances (units of
1 / SCALE) encoded as little-endian u16 bytes; see donors.donor.to_bytes.
Scores are (tau, parents at tau), lexicographic, lower is better.
"""

from typing import TypedDict

class ConstructResult(TypedDict):
    donors: list[int]
    """Selected donor indices, sorted."""
    tau: int
    """Largest distance from a parent to its nearest donor (scaled units)."""
    count: int
    """Number of parents at distance tau."""
    wall: float
    """Seconds taken."""

class OptimizeResult(TypedDict):
    donors: list[int]
    """Best run's donor indices, sorted."""
    tau: int
    """Best run's tau (scaled units)."""
    count: int
    """Best run's number of parents at tau."""
    sets: list[list[int]]
    """Distinct sorted donor sets at tau from all runs, at most max_sets, ordered by
    parents at tau; the first is `donors`."""
    init_tau: int
    """tau of the initial donors after swap descent."""
    iters: int
    """ALNS iterations summed over all runs."""
    run_iters: list[int]
    """Iterations of each run, in seed order."""
    wall: float
    """Seconds taken by the best run."""
    run_taus: list[int]
    """Final tau of each run, in seed order."""
    progress: list[tuple[int, int, int]]
    """Best run's (iteration, current tau, best tau), about every 0.5 s."""

def construct(
    dist: bytes, n: int, k: int, seeds: int = 10, seed: int = 0
) -> ConstructResult:
    """Best of `seeds` greedy + swap descent runs selecting k donors.

    Seed 0 starts empty; the others start from distinct random first donors
    (a prefix of one random order, so more seeds always include fewer-seed runs).

    Raises ValueError unless len(dist) == 2 * n * n, 1 <= k < n and seeds >= 1.
    """

def optimize(
    dist: bytes,
    n: int,
    init: list[int],
    max_iters: int = 1_000_000,
    patience: int = 100_000,
    time_limit_s: float = ...,
    seed: int = 0,
    runs: int = 1,
    max_sets: int = 1,
    config: dict[str, float] | None = None,
) -> OptimizeResult:
    """`runs` parallel ALNS searches from `init` (k = len(init)); returns the best.

    Run r uses seed `seed + r` and stops after `max_iters` iterations,
    `patience` iterations without a new best (0 = off), or `time_limit_s`
    seconds (default: no limit), whichever comes first. Each run keeps up to
    `max_sets` distinct donor sets at its best tau; `sets` merges those at the
    overall best tau. `config` overrides
    tuning: segment, react, score_best, score_better, score_accept, cooling,
    temp_factor, destroy_max.

    Raises ValueError unless len(dist) == 2 * n * n, init holds 1..n-1 indices
    below n, runs >= 1 and max_sets >= 1.
    """
