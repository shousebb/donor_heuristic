"""Donor selection as a discrete k-center problem.

Pick at most k donors from n recurrent parents so every parent lies within
genetic distance tau of a selected donor, minimizing tau.

Distances are scaled by `scale` (the [solver] scale setting, default SCALE)
and rounded, so tau is an integer and the answer is exact to 1 / scale.

The heuristics run in Rust (donors._core): a multi-start greedy + swap
construction, improved by parallel ALNS runs limited by iterations and
early-stopping patience. Z3 verifies the final cover. A CP-SAT model
(solve_cpsat) is available for lower bounds but is not part of the pipeline.

Usage: uv run donors [scenario.toml]
"""

import json
import math
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import z3
from ortools.sat.python import cp_model

from . import _core


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


SCALE = 100  # default: distances are solved in integer units of 1 / SCALE


def to_units(d: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Integer distances in units of 1 / scale; at most scale + 1 distinct values."""
    return np.rint(d * scale).astype(int)


def radius(d: np.ndarray, selected: list[int]) -> int:
    """tau of a selection: the largest distance from a parent to its nearest donor."""
    return int(d[:, selected].min(axis=1).max())


def nn_lower_bound(d: np.ndarray, k: int) -> int:
    """tau >= the (k+1)-th largest nearest-neighbour distance.

    Every parent that is not a donor is at least its nearest-neighbour
    distance from any donor, and at most k parents are donors.
    """
    nearest_other = np.where(np.eye(len(d), dtype=bool), np.iinfo(d.dtype).max, d).min(
        axis=1
    )
    return int(np.sort(nearest_other)[-(k + 1)])


def report(
    name: str,
    d: np.ndarray,
    selected: list[int],
    status: str,
    lower: float,
    seconds: float,
    scale: int,
) -> tuple[list[int], float, float]:
    """Print and return (selected donors, tau, proven lower bound on tau) in distance units.

    `d` and `lower` are in integer units of 1 / scale.
    """
    tau = d[:, selected].min(axis=1).max() / scale
    lower /= scale
    print(
        f"{name}: status={status} tau={tau:.4f} lower bound={lower:.4f} time={seconds:.1f}s"
    )
    return selected, tau, lower


def solve_window(
    d: np.ndarray,
    k: int,
    lower: int,
    upper: int,
    start: list[int],
    gap: float,
    time_limit: float,
) -> tuple[list[int], int, str]:
    """Radius-level model with tau restricted to [lower, upper].

    Levels are lower plus the distances in (lower, upper]; z[r] = 1 means
    tau >= levels[r]. For each parent j and each of its distances v in the
    window: tau reaches v, or a donor strictly closer than v is selected.
    Levels below the window are treated as switched on and levels above it
    are dropped, so the model's optimum is max(lower, optimal tau) as long as
    the optimum is <= upper (guaranteed: `start` has tau == upper).

    "A donor closer than v" is chained per parent: p_t implies p_(t-1) or a
    donor at the previous distance, so each donor appears once per parent.

    Returns (selected donors, proven lower bound on the model optimum, status).
    """
    n = len(d)
    levels = np.unique(
        np.r_[lower, d[(d > lower) & (d <= upper)]]
    )  # levels[0] == lower
    start_nearest = d[start].min(
        axis=0
    )  # each parent's distance to its nearest start donor
    start_tau = int(start_nearest.max())

    m = cp_model.CpModel()

    # Recurrent parent i is selected as a donor if x[i] = 1
    x = [m.new_bool_var(f"x{i}") for i in range(n)]
    # z[r] = 1 if tau >= levels[r]; z[0] is always on (tau >= lower)
    z = [m.new_bool_var(f"z{r}") for r in range(len(levels))]
    # tau in integer units of 1 / SCALE
    tau = m.new_int_var(lower, upper, "tau")

    # tau = lower + the level steps that are switched on
    m.add(
        tau
        == lower
        + sum(int(levels[r] - levels[r - 1]) * z[r] for r in range(1, len(levels)))
    )
    m.minimize(tau)

    # No more than k donors can be selected
    m.add(sum(x) <= k)

    # Levels switch on from the bottom up
    for r in range(2, len(levels)):
        m.add_implication(z[r], z[r - 1])

    for j in range(n):
        dist_j = d[:, j]
        window = np.unique(dist_j[(dist_j > lower) & (dist_j <= upper)])
        closer = None  # p: a donor strictly closer than the current v is selected
        previous_v = None
        for v in [
            *window,
            upper + 1,
        ]:  # upper + 1: parent j must be covered within upper
            new = (
                np.flatnonzero(dist_j < v)
                if previous_v is None
                else np.flatnonzero(dist_j == previous_v)
            )
            options = ([closer] if closer is not None else []) + [x[i] for i in new]
            if v > upper:
                m.add_bool_or(options)
                break
            p = m.new_bool_var("")
            m.add_bool_or([p.Not(), *options])  # p => closer donor selected
            m.add_bool_or(
                [z[int(np.searchsorted(levels, v))], p]
            )  # tau >= v, or closer donor
            m.add_hint(p, bool(start_nearest[j] < v))
            closer, previous_v = p, v

    # Warm start: hint every variable so the hint is a complete feasible solution
    chosen = set(start)
    for i in range(n):
        m.add_hint(x[i], i in chosen)
    for r in range(len(levels)):
        m.add_hint(z[r], bool(levels[r] <= start_tau))
    m.add_hint(tau, max(lower, start_tau))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.relative_gap_limit = gap
    solver.parameters.log_search_progress = True
    solver.parameters.num_workers = 8
    status = solver.solve(m)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return start, lower, solver.status_name(status)
    selected = [i for i in range(n) if solver.value(x[i])]
    return (
        selected,
        int(np.ceil(solver.best_objective_bound)),
        solver.status_name(status),
    )


def solve_cpsat(
    d: np.ndarray,
    k: int,
    start: list[int],
    gap: float = 0.01,
    window: float = 0.05,
    time_limit: float = 120,
    scale: int = SCALE,
) -> tuple[list[int], float, float]:
    """Improve `start` and prove a lower bound, searching just below its tau.

    Solves the radius-level model on the window [upper * (1 - window), upper],
    where upper is the best tau found so far. If the optimum turns out to be
    at or below the window, the window is doubled and solved again. Stops
    each solve at relative gap `gap`. Returns (selected donors, tau, proven
    lower bound on tau), to 1 / scale.
    """
    t0 = time.perf_counter()
    d = to_units(d, scale)
    floor = nn_lower_bound(d, k)
    best, upper = list(start), radius(d, start)
    while True:
        lower = max(int(upper * (1 - window)), floor - 1)
        print(f"cp-sat window: [{lower / scale:.4f}, {upper / scale:.4f}]")
        selected, bound, status = solve_window(
            d, k, lower, upper, best, gap, time_limit
        )
        found = radius(d, selected)
        if found < upper:
            best, upper = selected, found
        if bound > lower or lower < floor:
            break  # bound is a valid lower bound on the optimal tau
        if found > lower:
            bound = floor  # stopped without settling the window: only the trivial bound holds
            break
        window *= 2  # optimum is at or below the window: widen it downwards
    return report(
        "cp-sat", d, best, status, max(bound, floor), time.perf_counter() - t0, scale
    )


def to_bytes(d: np.ndarray) -> bytes:
    """Scaled distances as the little-endian u16 buffer the Rust kernel reads."""
    assert d.max() <= np.iinfo(np.uint16).max, "scaled distances must fit in u16"
    return d.astype("<u2").tobytes()


def solve_heuristic(
    d: np.ndarray, k: int, seeds: int = 10, seed: int = 0, scale: int = SCALE
) -> tuple[list[int], float, float]:
    """Rust multi-start greedy + swap descent. Heuristic, no lower bound.

    Seed 0 is the plain greedy; the others start from distinct random first
    donors. Ties on tau are broken by fewer parents at tau.
    """
    d = to_units(d, scale)
    result = _core.construct(to_bytes(d), len(d), k, seeds, seed)
    status = f"best of {seeds} seeds"
    return report("heuristic", d, result["donors"], status, 0, result["wall"], scale)


def solve_alns(
    d: np.ndarray,
    start: list[int],
    max_iters: int = 1_000_000,
    patience: int = 100_000,
    time_limit: float = math.inf,
    runs: int = 4,
    seed: int = 0,
    scale: int = SCALE,
    max_sets: int = 1,
) -> tuple[list[int], float, float, list[list[int]]]:
    """Rust ALNS from `start` (k = len(start)): `runs` parallel searches, best kept.

    Each run stops after `max_iters` iterations, `patience` iterations without
    a new best (0 = off), or `time_limit` seconds, whichever comes first.
    Destroys a few donors per iteration (random, nearest the bottleneck parent,
    or related), repairs greedily with swap descent, and accepts by simulated
    annealing. Heuristic, no lower bound.

    Returns (selected donors, tau, lower bound 0, sets): `sets` holds up to
    `max_sets` distinct donor sets the search found at that same tau, fewest
    parents at tau first; sets[0] is `selected`.
    """
    d = to_units(d, scale)
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
    run_taus = ", ".join(f"{t / scale:.4f}" for t in result["run_taus"])
    status = (
        f"run iterations {result['run_iters']}, run taus [{run_taus}], "
        f"sets at tau {len(result['sets'])}"
    )
    selected, tau, lower = report(
        "alns", d, result["donors"], status, 0, result["wall"], scale
    )
    return selected, tau, lower, result["sets"]


def verify(
    d: np.ndarray, selected: list[int], tau: float, k: int, scale: int = SCALE
) -> bool:
    """Z3 check: at most k donors, and every parent has a selected donor within tau.

    Compares in the same integer units of 1 / scale that the solvers use.
    """
    d = to_units(d, scale)
    t = z3.Int("tau")
    s = z3.Solver()
    s.add(t == round(tau * scale))
    s.add(len(set(selected)) <= k)
    for j in range(len(d)):
        s.add(z3.Or([int(d[i, j]) <= t for i in selected]))
    return s.check() == z3.sat


@dataclass(frozen=True)
class Data:
    """Synthetic data: similarity ~ N(mu, sigma) clipped to [0, 1]."""

    parents: int = 1000
    mu: float = 0.7
    sigma: float = 0.08
    seed: int = 42


@dataclass(frozen=True)
class Solver:
    """Heuristic settings; each ALNS run stops at the first limit reached."""

    donors: int = 10
    seeds: int = 10
    runs: int = 4
    max_iters: int = 1_000_000
    patience: int = 100_000
    time_limit: float = math.inf
    seed: int = 0
    scale: int = SCALE  # distances solved in integer units of 1 / scale (<= 65 535)
    max_sets: int = 10  # distinct donor sets to report at the best tau


@dataclass(frozen=True)
class Scenario:
    """A scenario file: [data] and [solver] tables; omitted keys use defaults."""

    data: Data = field(default_factory=Data)
    solver: Solver = field(default_factory=Solver)

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        """Read a TOML scenario; unknown keys raise TypeError."""
        cfg = tomllib.loads(path.read_text())
        return cls(Data(**cfg.get("data", {})), Solver(**cfg.get("solver", {})))


def write_output(
    scenario_path: Path,
    scenario: Scenario,
    result: dict[str, object],
    out_root: Path | None = None,
) -> Path:
    """Write the scenario and result to <out_root>/<scenario stem>_<YYYYMMDD>/result.json.

    out_root defaults to ./outputs.

    A rerun on the same day overwrites it. An infinite time_limit is written as null.
    """
    solver = {
        key: None if value == math.inf else value
        for key, value in asdict(scenario.solver).items()
    }
    out_dir = (
        out_root or Path("outputs")
    ) / f"{scenario_path.stem}_{datetime.now().astimezone():%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "result.json"
    payload = {
        "scenario": {
            "file": str(scenario_path),
            "data": asdict(scenario.data),
            "solver": solver,
        },
        "result": result,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "scenario.toml")
    scenario = Scenario.load(path)
    data, solver = scenario.data, scenario.solver
    print(f"scenario {path}: {data}, {solver}")

    distances = get_distances(data.parents, data.mu, data.sigma, (0, 1), data.seed)
    start, start_tau, _ = solve_heuristic(
        distances, solver.donors, seeds=solver.seeds, scale=solver.scale
    )
    _, tau, _, sets = solve_alns(
        distances,
        start,
        max_iters=solver.max_iters,
        patience=solver.patience,
        time_limit=solver.time_limit,
        runs=solver.runs,
        seed=solver.seed,
        scale=solver.scale,
        max_sets=solver.max_sets,
    )
    print(f"alns tau is {1 - tau / start_tau:.1%} below the heuristic")
    d = to_units(distances, solver.scale)
    at_tau = [
        int((d[:, s].min(axis=1) == round(tau * solver.scale)).sum()) for s in sets
    ]
    print(f"{len(sets)} donor sets at tau={tau:.4f}, fewest parents at tau first:")
    for donors, count in zip(sets, at_tau, strict=True):
        print(f"  {donors} ({count} parents at tau)")
        assert verify(distances, donors, tau, solver.donors, solver.scale), (
            f"z3: donor set {donors} leaves a parent uncovered"
        )
    print("z3: every set covers all parents within tau")

    out = write_output(
        path,
        scenario,
        {
            "heuristic_tau": start_tau,
            "tau": tau,
            "sets": [
                {"donors": donors, "parents_at_tau": count}
                for donors, count in zip(sets, at_tau, strict=True)
            ],
        },
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
