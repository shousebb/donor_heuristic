import contextlib
import io
import time

import numpy as np
from donors import donor

n = 2000
d = donor.get_distances(n=n, mu=0.7, sigma=0.08, bounds=(0, 1))
D = donor.scale(d)
print(f"levels total: {len(np.unique(D))}")
for k in (5, 20):
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sel, tau, _ = donor.solve_heuristic(d, k)
    U = round(tau * donor.SCALE)
    below = (D <= U).sum(axis=0)
    print(
        f"k={k}: heuristic tau={tau:.4f} in {time.perf_counter() - t:.1f}s; donors within U per parent: mean {below.mean():.0f} max {below.max()}"
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
        print(
            f"   window {w:.0%}: L={L / donor.SCALE:.4f} levels={len(np.unique(D[(D > L) & (D <= U)]))} "
            f"constraints={sum(len(np.unique(D[:, j][(D[:, j] > L) & (D[:, j] <= U)])) for j in range(n))} literals={lits / 1e6:.1f}M"
        )
