"""CP-SAT radius-level model for lower bounds (not part of the CLI pipeline).

Takes the n x n matrix of integer distances from distances.to_units.
"""

import time

import numpy as np
from ortools.sat.python import cp_model

from .distances import parents_at_tau, radius
from .solution import Solution


def nn_lower_bound(d: np.ndarray, k: int) -> int:
    """tau >= the (k+1)-th largest nearest-neighbour distance.

    Every parent that is not a donor is at least its nearest-neighbour
    distance from any donor, and at most k parents are donors.
    """
    nearest_other = np.where(np.eye(len(d), dtype=bool), np.iinfo(d.dtype).max, d).min(
        axis=1
    )
    return int(np.sort(nearest_other)[-(k + 1)])


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
    # tau in integer units of 1 / scale
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
) -> Solution:
    """Improve `start` and prove a lower bound, searching just below its tau.

    Solves the radius-level model on the window [upper * (1 - window), upper],
    where upper is the best tau found so far. If the optimum turns out to be
    at or below the window, the window is doubled and solved again. Stops
    each solve at relative gap `gap`.
    """
    t0 = time.perf_counter()
    floor = nn_lower_bound(d, k)
    best, upper = list(start), radius(d, start)
    while True:
        lower = max(int(upper * (1 - window)), floor - 1)
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
    count = parents_at_tau(d, best)
    return Solution(
        donors=best,
        tau=upper,
        count=count,
        lower=max(bound, floor),
        seconds=time.perf_counter() - t0,
        status=status,
        sets=[best],
        counts=[count],
    )
