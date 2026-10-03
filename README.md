# Donor Optimization

## Problem

Given a list of recurrent parents and the genetic distance between them, select at
most `k` donors from the recurrent parents so that every recurrent parent is within
genetic distance `tau` of a selected donor, minimizing `tau`. This is the discrete
k-center problem.

Distances are solved in integer units of `1 / SCALE` (set in `donor.py`; 100 gives
0.01, 10 000 gives 1e-4), so `tau` is exact to that resolution. A coarser `SCALE` merges
near-equal distances, which shrinks the CP-SAT model and the Z3 checks.

## Example data

Synthetic genetic distances are `1 - similarity`, with similarity drawn from
`N(mu, sigma)`, clipped to `[0, 1]`, symmetric, and 1 on the diagonal:

- 1000 recurrent parents by default (`--parents`; up to about 2000)
- at most 10 donors (`--donors`; 5 to 20 expected)
- similarity mean 0.7 (`--mu`), standard deviation 0.08 (`--sigma`)

Independent random distances are the hardest case for proving optimality: there is no
cluster structure, so many donor sets are nearly as good as the best.

## Approach

1. **Construction (Rust):** greedy donor selection plus swap descent from `--seeds`
   starts; the best is kept.
2. **ALNS (Rust):** `--runs` parallel adaptive large neighbourhood searches for
   `--alns-time` seconds. Each iteration removes 1 to 3 donors (random, nearest the
   parent at `tau`, or mutually close), repairs greedily, runs swap descent, and
   accepts by simulated annealing. This gives the upper bound.
3. **CP-SAT (optional):** a radius-level model restricted to a window just below the
   best `tau`, warm-started from it and stopped at relative gap `--gap`. It can
   improve the cover and prove a lower bound. Skip it with `--cpsat-time 0`.
4. **Z3:** verifies every cover, checks CP-SAT's lower bound, then raises the lower
   bound by binary search (`--z3-time` in total, `--z3-check-time` per check).

The run ends with `optimal tau in [lower, upper], gap ...`.

## Usage

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) and a Rust toolchain; `uv`
builds the Rust extension automatically.

```bash
uv run donors                    # n = 1000, k = 10
uv run donors --cpsat-time 0     # heuristics + Z3 lower bound only
uv run donors --help             # all options
```

From Python:

```python
from donors.donor import get_distances, solve_heuristic, solve_alns, z3_lower_bound

d = get_distances(n=2000, mu=0.7, sigma=0.08, bounds=(0, 1))
start, _, _ = solve_heuristic(d, k=10)
donors, tau, _ = solve_alns(d, start, time_limit=60, runs=4)
lower, better = z3_lower_bound(d, 10, 0.0, tau)
```

## Results (n = 2000, k = 10)

| method | tau |
|---|---|
| Python multi-start greedy + swap, 1000 starts | 0.2667 |
| Rust ALNS, 60 s, single run | 0.2628 to 0.2660 |

The lower bound is the weak side. Bounds based on the linear relaxation can prove at
most about 0.2066 on this data, so expect a gap above 20%. On 300 parents, the Z3
search proved `tau >= 0.1423` against a cover at 0.2319.

## Layout

```
Cargo.toml, src/lib.rs        Rust extension donors._core: construct, optimize
python/donors/_core.pyi       type stubs for the extension
python/donors/donor.py        data, CP-SAT, Z3, wrappers and CLI (donors)
tests/test_donors.py          pytest suite
reference/donor.py            earlier SCIP big-M model (reference only)
```

## Development

```bash
uv run pytest                             # tests
uv run ruff check && uv run ruff format   # lint and format
uv run ty check && uv run basedpyright    # type checks
cargo fmt && cargo clippy -- -D warnings  # Rust
```

`uv` rebuilds the extension when `Cargo.toml` or `src/**/*.rs` change (see
`[tool.uv] cache-keys` in `pyproject.toml`).
