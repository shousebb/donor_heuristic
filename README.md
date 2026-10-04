# Donor Optimization

## Problem

Given a list of recurrent parents and the genetic distance between them, select at
most `k` donors from the recurrent parents so that every recurrent parent is within
genetic distance `tau` of a selected donor, minimizing `tau`. This is the discrete
k-center problem.

Distances are solved in integer units of `1 / scale`, set by `[solver] scale` in the
scenario (default 100 gives 0.01; 10 000 gives 1e-4; at most 65 535), so `tau` is exact
to that resolution. A coarser scale merges near-equal distances into fewer distinct
values.

## Example data

Synthetic genetic distances are `1 - similarity`, with similarity drawn from
`N(mu, sigma)`, clipped to `[0, 1]`, symmetric, and 1 on the diagonal:

- 1000 recurrent parents by default (`[data] parents`; up to about 2000)
- at most 10 donors (`[solver] donors`; 5 to 20 expected)
- similarity mean 0.7 (`mu`), standard deviation 0.08 (`sigma`)

Independent random distances are the hardest case for proving optimality: there is no
cluster structure, so many donor sets are nearly as good as the best.

## Approach

1. **Construction (Rust):** greedy donor selection plus swap descent from `seeds`
   starts; the best is kept.
2. **ALNS (Rust):** `runs` parallel adaptive large neighbourhood searches. Each
   iteration removes 1 to 3 donors (random, nearest the parent at `tau`, or mutually
   close), repairs greedily, runs swap descent, and accepts by simulated annealing.
   Each run stops after `max_iters` iterations (default 1 000 000) or `patience`
   iterations without a new best (default 100 000), whichever comes first; an
   optional `time_limit` caps its seconds. Every run also keeps the distinct donor
   sets it meets at its best `tau`; those at the overall best `tau` are merged and up to
   `max_sets` are reported, fewest parents at `tau` first.
3. **Z3:** verifies that each reported set uses at most `donors` donors and covers
   every parent within `tau`.

The heuristics give an upper bound only. A CP-SAT radius-level model
(`solve_cpsat`) remains in `donor.py` for lower bounds but is not part of the
pipeline.

## Usage

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) and a Rust toolchain; `uv`
builds the Rust extension automatically.

```bash
uv run donors                    # reads ./scenario.toml
uv run donors my_scenario.toml   # another scenario
```

Each run writes the scenario and its result (heuristic tau, final tau, and each donor
set with its number of parents at tau) to `outputs/<scenario name>_<YYYYMMDD>/result.json`,
e.g. `outputs/scenario_20261004/result.json`; a rerun on the same day overwrites it.
`outputs/` is git-ignored.

A scenario has a `[data]` and a `[solver]` table; omitted keys use the defaults and
unknown keys are an error. See `scenario.toml`:

```toml
[data]
parents = 1000   # recurrent parents n
mu = 0.7         # mean similarity; distance = 1 - similarity
sigma = 0.08     # similarity standard deviation (clipped to [0, 1])
seed = 42        # synthetic data seed

[solver]
donors = 10            # maximum donors k
seeds = 10             # greedy construction starts
runs = 4               # parallel ALNS runs, best kept
max_iters = 1_000_000  # iterations per ALNS run
patience = 100_000     # stop a run after this many iterations without a new best
seed = 0               # ALNS seed (run r uses seed + r)
scale = 100            # distances solved in integer units of 1 / scale (100 -> 0.01; max 65_535)
max_sets = 10          # distinct donor sets to report at the best tau
# time_limit = 600     # optional seconds per run (default: no limit)
```

From Python:

```python
from donors.donor import get_distances, solve_alns, solve_heuristic, verify

d = get_distances(n=2000, mu=0.7, sigma=0.08, bounds=(0, 1))
start, _, _ = solve_heuristic(d, k=10)
donors, tau, _, sets = solve_alns(
    d, start, max_iters=1_000_000, patience=100_000, max_sets=10
)
assert all(verify(d, s, tau, k=10) for s in sets)  # sets[0] == donors
```

## Results (n = 2000, k = 10)

| method | tau |
|---|---|
| Python multi-start greedy + swap, 1000 starts | 0.2667 |
| Rust ALNS, 60 s, single run | 0.2628 to 0.2660 |

Measured at `scale = 10_000` before the iteration and patience limits. Lower bounds
are weak on this data: bounds based on the linear relaxation can prove at most about
0.2066, so the gap to the best cover is above 20%.

## Layout

```
Cargo.toml, src/lib.rs        Rust extension donors._core: construct, optimize
python/donors/_core.pyi       type stubs for the extension
python/donors/donor.py        data, scenario, wrappers, Z3 check, CLI (donors)
scenario.toml                 default scenario
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
