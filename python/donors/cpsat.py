"""CP-SAT models on the n x n matrix of integer distances from distances.to_units.

check_tau / descend: does a cover within a given tau exist? Used by the CLI to
confirm the best cover and, optionally, to search downwards from the ALNS tau.
solve_cpsat: radius-level model for lower bounds (not part of the CLI pipeline).
"""

import time
from dataclasses import dataclass

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
    log: bool = False,
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
    solver.parameters.log_search_progress = log
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
    log: bool = False,
) -> Solution:
    """Improve `start` and prove a lower bound, searching just below its tau.

    Solves the radius-level model on the window [upper * (1 - window), upper],
    where upper is the best tau found so far. If the optimum turns out to be
    at or below the window, the window is doubled and solved again. Stops
    each solve at relative gap `gap`. `log` prints CP-SAT's search progress.
    """
    t0 = time.perf_counter()
    floor = nn_lower_bound(d, k)
    best, upper = list(start), radius(d, start)
    while True:
        lower = max(int(upper * (1 - window)), floor - 1)
        selected, bound, status = solve_window(
            d, k, lower, upper, best, gap, time_limit, log
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


@dataclass(frozen=True)
class Check:
    """One CP-SAT check: can at most k donors cover every parent within tau?"""

    tau: int
    result: str  # "feasible" (donors found), "infeasible" or "unknown" (timed out)
    seconds: float  # time spent solving, excluding model construction
    donors: list[int] | None  # a cover at tau when feasible


def check_tau(
    d: np.ndarray,
    k: int,
    tau: int,
    timeout: float,
    donors: list[int] | None = None,
    hint: list[int] | None = None,
    log: bool = False,
) -> Check:
    """Search all selections of at most k donors for a cover within tau, with CP-SAT.

    x[i] selects parent i as a donor; each parent needs a selected donor within
    tau. Gives up after `timeout` seconds with result "unknown". Given `donors`,
    checks only that exact selection. `hint` warm-starts the search; `log` prints
    CP-SAT's search progress.
    """
    n = len(d)
    m = cp_model.CpModel()
    x = [m.new_bool_var(f"x{i}") for i in range(n)]
    m.add(sum(x) <= k)
    for j in range(n):
        m.add_bool_or([x[i] for i in np.flatnonzero(d[:, j] <= tau)])
    if donors is not None:
        chosen = set(donors)
        for i in range(n):
            m.add(x[i] == int(i in chosen))
    if hint is not None:
        hinted = set(hint)
        for i in range(n):
            m.add_hint(x[i], i in hinted)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = log
    t0 = time.perf_counter()
    status = solver.solve(m)
    seconds = time.perf_counter() - t0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        found = [i for i in range(n) if solver.value(x[i])]
        return Check(tau, "feasible", seconds, found)
    result = "infeasible" if status == cp_model.INFEASIBLE else "unknown"
    return Check(tau, result, seconds, None)


def descend(
    d: np.ndarray,
    k: int,
    tau: int,
    timeout: float = 30,
    donors: list[int] | None = None,
    log: bool = False,
) -> list[Check]:
    """Check tau, tau - 1, ... with CP-SAT until one is not feasible or exceeds `timeout`.

    If the last check is infeasible, its tau + 1 is proven optimal; every
    feasible check gives a cover at its tau. A check that runs past `timeout`
    seconds ends the search, normally with result "unknown". `donors`, a known
    cover at tau, makes the first check confirm it instead of searching (an
    infeasible first check then means `donors` is not a cover); each later
    check is warm-started from the last cover found. `log` prints CP-SAT's
    search progress for every check.
    """
    checks: list[Check] = []
    last = donors
    for t in range(tau, -1, -1):
        check = check_tau(d, k, t, timeout, donors if t == tau else None, last, log)
        checks.append(check)
        if check.result != "feasible" or check.seconds > timeout:
            break
        last = check.donors
    return checks
