"""Donor selection as a discrete k-center problem.

Pick at most k donors from n recurrent parents so every parent lies within
genetic distance tau of a selected donor, minimizing tau.

Distances are scaled by SCALE and rounded, so tau is an integer and the
answer is exact to 1 / SCALE.

The heuristics run in Rust (donors._core): a multi-start greedy + swap
construction, improved by parallel ALNS runs (upper bound). CP-SAT then
solves a radius-level model restricted to a window just below it, stopping at
a relative gap, to improve the cover and prove a lower bound (optional). Z3
verifies each cover, checks CP-SAT's bound, and raises the lower bound by
binary search.

Usage: uv run donors [--help]
"""

import argparse
import time

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


SCALE = 100  # distances are solved in integer units of 1 / SCALE


def scale(d: np.ndarray) -> np.ndarray:
    """Integer distances in units of 1 / SCALE; at most SCALE + 1 distinct values."""
    return np.rint(d * SCALE).astype(int)


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
) -> tuple[list[int], float, float]:
    """Print and return (selected donors, tau, proven lower bound on tau) in distance units."""
    tau = d[:, selected].min(axis=1).max() / SCALE
    lower /= SCALE
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
) -> tuple[list[int], float, float]:
    """Improve `start` and prove a lower bound, searching just below its tau.

    Solves the radius-level model on the window [upper * (1 - window), upper],
    where upper is the best tau found so far. If the optimum turns out to be
    at or below the window, the window is doubled and solved again. Stops
    each solve at relative gap `gap`. Returns (selected donors, tau, proven
    lower bound on tau), to 1 / SCALE.
    """
    t0 = time.perf_counter()
    d = scale(d)
    floor = nn_lower_bound(d, k)
    best, upper = list(start), radius(d, start)
    while True:
        lower = max(int(upper * (1 - window)), floor - 1)
        print(f"cp-sat window: [{lower / SCALE:.4f}, {upper / SCALE:.4f}]")
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
        "cp-sat", d, best, status, max(bound, floor), time.perf_counter() - t0
    )


def to_bytes(d: np.ndarray) -> bytes:
    """Scaled distances as the little-endian u16 buffer the Rust kernel reads."""
    assert d.max() <= np.iinfo(np.uint16).max, "scaled distances must fit in u16"
    return d.astype("<u2").tobytes()


def solve_heuristic(
    d: np.ndarray, k: int, seeds: int = 10, seed: int = 0
) -> tuple[list[int], float, float]:
    """Rust multi-start greedy + swap descent. Heuristic, no lower bound.

    Seed 0 is the plain greedy; the others start from distinct random first
    donors. Ties on tau are broken by fewer parents at tau.
    """
    d = scale(d)
    result = _core.construct(to_bytes(d), len(d), k, seeds, seed)
    return report(
        "heuristic", d, result["donors"], f"best of {seeds} seeds", 0, result["wall"]
    )


def solve_alns(
    d: np.ndarray,
    start: list[int],
    time_limit: float = 60,
    runs: int = 4,
    seed: int = 0,
) -> tuple[list[int], float, float]:
    """Rust ALNS from `start` (k = len(start)): `runs` parallel searches, best kept.

    Destroys a few donors per iteration (random, nearest the bottleneck parent,
    or related), repairs greedily with swap descent, and accepts by simulated
    annealing. Heuristic, no lower bound.
    """
    d = scale(d)
    result = _core.optimize(
        to_bytes(d), len(d), [int(i) for i in start], time_limit, seed, runs
    )
    run_taus = ", ".join(f"{t / SCALE:.4f}" for t in result["run_taus"])
    status = f"{result['iters']} iterations, run taus [{run_taus}]"
    return report("alns", d, result["donors"], status, 0, result["wall"])


def verify(d: np.ndarray, selected: list[int], tau: float, k: int) -> bool:
    """Z3 check: at most k donors, and every parent has a selected donor within tau.

    Compares in the same integer units of 1 / SCALE that the solvers use.
    """
    d = scale(d)
    t = z3.Int("tau")
    s = z3.Solver()
    s.add(t == round(tau * SCALE))
    s.add(len(set(selected)) <= k)
    for j in range(len(d)):
        s.add(z3.Or([int(d[i, j]) <= t for i in selected]))
    return s.check() == z3.sat


def verify_lower(d: np.ndarray, lower: float, k: int, time_limit: float = 300) -> str:
    """Z3 check of a lower bound: no k donors cover every parent within lower - 1 / SCALE.

    Returns "proven" (unsat), "refuted" (a better cover exists) or "unknown" (timed out).
    """
    return z3_cover_within(scale(d), round(lower * SCALE) - 1, k, time_limit)[0]


def z3_cover_within(
    d: np.ndarray, within: int, k: int, time_limit: float
) -> tuple[str, list[int] | None]:
    """Z3: can k donors cover every parent within scaled distance `within`?

    Returns ("proven", None) if not (so optimal tau > within), ("refuted",
    donors) with such a cover, or ("unknown", None) on timeout.
    """
    x = [z3.Bool(f"x{i}") for i in range(len(d))]  # donor i is selected
    s = z3.Solver()
    s.set("timeout", int(time_limit * 1000))
    s.add(z3.AtMost(*x, k))
    for j in range(len(d)):
        s.add(z3.Or([x[i] for i in np.flatnonzero(d[:, j] <= within)]))
    result = s.check()
    if result == z3.unsat:
        return "proven", None
    if result == z3.sat:
        model = s.model()
        return "refuted", [i for i in range(len(d)) if z3.is_true(model.eval(x[i]))]
    return "unknown", None


def z3_lower_bound(
    d: np.ndarray,
    k: int,
    lower: float,
    upper: float,
    time_limit: float = 300,
    check_limit: float = 30,
) -> tuple[float, list[int] | None]:
    """Raise a proven lower bound on tau by binary search with Z3.

    Starts from `lower` (already proven, raised to the nearest-neighbour
    bound) and the cover tau `upper`. Each step asks whether k donors can
    cover every parent below a probe: proven impossible raises the bound;
    a cover found lowers the upper bound; a timeout searches lower probes.
    Each check gets at most `check_limit` seconds, all of them `time_limit`.

    Returns (proven lower bound on tau, better cover found or None).
    """
    d = scale(d)
    lo = max(round(lower * SCALE), nn_lower_bound(d, k))  # optimal tau >= lo
    hi = round(upper * SCALE)  # probes stay <= hi
    better = None
    deadline = time.perf_counter() + time_limit
    while lo < hi and (remaining := deadline - time.perf_counter()) > 0:
        probe = (lo + hi + 1) // 2  # try to prove optimal tau >= probe
        t0 = time.perf_counter()
        status, donors = z3_cover_within(d, probe - 1, k, min(check_limit, remaining))
        print(
            f"z3: tau >= {probe / SCALE:.4f}? {status} ({time.perf_counter() - t0:.1f}s)"
        )
        if status == "proven":
            lo = probe
        elif donors is not None:
            better, hi = donors, radius(d, donors)
        else:
            hi = probe - 1
    return lo / SCALE, better


class Args(argparse.Namespace):
    """Typed command-line options and their defaults."""

    parents: int = 1000
    donors: int = 10
    mu: float = 0.7
    sigma: float = 0.08
    data_seed: int = 42
    seeds: int = 10
    alns_time: float = 60.0
    runs: int = 4
    cpsat_time: float = 120.0
    gap: float = 0.01
    z3_time: float = 300.0
    z3_check_time: float = 30.0


OPTIONS = {
    "parents": "number of recurrent parents n",
    "donors": "maximum donors k",
    "mu": "mean similarity",
    "sigma": "similarity standard deviation",
    "data_seed": "seed for the synthetic data",
    "seeds": "greedy construction starts",
    "alns_time": "seconds per ALNS run",
    "runs": "parallel ALNS runs",
    "cpsat_time": "seconds per CP-SAT window (0 skips CP-SAT)",
    "gap": "CP-SAT relative gap limit",
    "z3_time": "seconds for the Z3 lower-bound search",
    "z3_check_time": "seconds per Z3 check",
}


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    for dest, text in OPTIONS.items():
        default: int | float = getattr(Args, dest)
        flag = "--" + dest.replace("_", "-")
        p.add_argument(flag, type=type(default), help=f"{text} (default {default})")
    args = p.parse_args(namespace=Args())

    k = args.donors
    distances = get_distances(args.parents, args.mu, args.sigma, (0, 1), args.data_seed)

    def check(name: str, selected: list[int], tau: float) -> None:
        print(f"donors={sorted(selected)}, tau={tau:.4f}")
        assert verify(distances, selected, tau, k), (
            f"z3: {name} leaves a parent uncovered"
        )
        print("z3: all parents covered within tau")

    heuristic, heuristic_tau, _ = solve_heuristic(distances, k, seeds=args.seeds)
    check("heuristic", heuristic, heuristic_tau)
    improved, improved_tau, _ = solve_alns(
        distances, heuristic, time_limit=args.alns_time, runs=args.runs
    )
    check("alns", improved, improved_tau)
    print(f"alns tau is {1 - improved_tau / heuristic_tau:.1%} below the heuristic")
    selected, tau, lower = improved, improved_tau, 0.0

    if args.cpsat_time > 0:
        selected, tau, lower = solve_cpsat(
            distances, k, start=improved, gap=args.gap, time_limit=args.cpsat_time
        )
        check("cp-sat", selected, tau)
        result = verify_lower(distances, lower, k, time_limit=args.z3_check_time)
        assert result != "refuted", "z3: a cover below the cp-sat lower bound exists"
        print(f"z3: cp-sat lower bound {lower:.4f} {result}")

    lower, better = z3_lower_bound(
        distances,
        k,
        lower,
        tau,
        time_limit=args.z3_time,
        check_limit=args.z3_check_time,
    )
    if better is not None:
        selected, tau = better, radius(scale(distances), better) / SCALE
        check("z3", selected, tau)
    print(f"optimal tau in [{lower:.4f}, {tau:.4f}], gap {1 - lower / tau:.2%}")


if __name__ == "__main__":
    main()
