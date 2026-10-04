"""Size of the CP-SAT radius-level model for tau windows below the heuristic tau.

Usage: uv run python scripts/measure_windows.py
"""

import time

import numpy as np
from donors.distances import SCALE, get_distances, to_units
from donors.heuristics import solve_heuristic

n = 2000
D = to_units(get_distances(n=n, mu=0.7, sigma=0.08, bounds=(0, 1)))
print(f"levels total: {len(np.unique(D))}")
for k in (5, 20):
    t = time.perf_counter()
    U = solve_heuristic(D, k).tau
    below = (D <= U).sum(axis=0)
    print(
        f"k={k}: heuristic tau={U / SCALE:.4f} in {time.perf_counter() - t:.1f}s; donors within U per parent: mean {below.mean():.0f} max {below.max()}"
    )
    for w in (0.02, 0.05, 0.10):
        L = int(U * (1 - w))
        in_win = ((D > L) & (D <= U)).sum(axis=0)
        # direct encoding literals: for each parent, each window distance v, count donors closer than v
        lits = 0
        for j in range(n):
            col = np.sort(D[:, j][D[:, j] <= U])
            vs = np.unique(col[col > L])
            lits += np.searchsorted(col, vs).sum()
        levels = len(np.unique(D[(D > L) & (D <= U)]))
        constraints = sum(
            len(np.unique(D[:, j][(D[:, j] > L) & (D[:, j] <= U)])) for j in range(n)
        )
        print(
            f"   window {w:.0%}: L={L / SCALE:.4f} levels={levels} "
            + f"constraints={constraints} literals={lits / 1e6:.1f}M"
        )
