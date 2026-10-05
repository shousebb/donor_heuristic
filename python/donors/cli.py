"""Command line: solve a scenario and save the result.

Usage: uv run donors [scenario.toml]

Builds synthetic distances, runs the Rust heuristics (greedy + swap
construction, then parallel ALNS), checks every donor set found at the best
tau (a CP-SAT model confirms the best one), optionally searches for a lower
tau with CP-SAT (cpsat_search), and writes the scenario and result to
outputs/<scenario stem>_<YYYYMMDD>/result_<HHMMSS>.json.
"""

import json
import math
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .cpsat import Check, descend
from .distances import SCALE, get_distances, to_units
from .heuristics import solve_alns, solve_heuristic
from .verify import covers, verify


def _require(checks: dict[str, bool]) -> None:
    """Raise ValueError naming every failed check."""
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"invalid scenario: need {', '.join(failed)}")


@dataclass(frozen=True)
class Data:
    """Synthetic data: similarity ~ N(mu, sigma) clipped to [0, 1]."""

    parents: int = 1000
    mu: float = 0.7
    sigma: float = 0.08
    seed: int = 42

    def __post_init__(self) -> None:
        _require({"parents >= 2": self.parents >= 2, "sigma >= 0": self.sigma >= 0})


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
    cpsat_search: bool = False  # after ALNS, lower tau by 1 with CP-SAT until one fails
    cpsat_timeout: float = 30  # stop the CP-SAT search when a check takes longer (s)
    cpsat_log: bool = False  # print CP-SAT's search progress during the search

    def __post_init__(self) -> None:
        _require(
            {
                "donors >= 1": self.donors >= 1,
                "seeds >= 1": self.seeds >= 1,
                "runs >= 1": self.runs >= 1,
                "max_iters >= 1": self.max_iters >= 1,
                "patience >= 0": self.patience >= 0,
                "time_limit > 0": self.time_limit > 0,
                "seed >= 0": self.seed >= 0,
                "1 <= scale <= 65535": 1 <= self.scale <= 65_535,
                "max_sets >= 1": self.max_sets >= 1,
                "cpsat_timeout > 0": self.cpsat_timeout > 0,
            }
        )


@dataclass(frozen=True)
class Scenario:
    """A scenario file: [data] and [solver] tables; omitted keys use defaults."""

    data: Data = field(default_factory=Data)
    solver: Solver = field(default_factory=Solver)

    def __post_init__(self) -> None:
        _require({"donors < parents": self.solver.donors < self.data.parents})

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        """Read a TOML scenario; unknown keys raise TypeError, bad values ValueError."""
        cfg = tomllib.loads(path.read_text())
        return cls(Data(**cfg.get("data", {})), Solver(**cfg.get("solver", {})))


def write_output(
    scenario_path: Path,
    scenario: Scenario,
    result: dict[str, object],
    out_root: Path | None = None,
) -> Path:
    """Write the scenario and result to a new JSON file and return its path.

    The file is <out_root>/<scenario stem>_<YYYYMMDD>/result_<HHMMSS>.json
    (out_root defaults to ./outputs); a numeric suffix keeps it from replacing
    an earlier file. An infinite time_limit is written as null.
    """
    now = datetime.now().astimezone()
    out_dir = (out_root or Path("outputs")) / f"{scenario_path.stem}_{now:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    solver = {
        key: None if value == math.inf else value
        for key, value in asdict(scenario.solver).items()
    }
    payload = {
        "scenario": {
            "file": str(scenario_path.resolve()),
            "data": asdict(scenario.data),
            "solver": solver,
        },
        "time": now.isoformat(timespec="seconds"),
        "result": result,
    }
    text = json.dumps(payload, indent=2) + "\n"
    for attempt in range(1, 1000):
        suffix = "" if attempt == 1 else f"_{attempt}"
        out = out_dir / f"result_{now:%H%M%S}{suffix}.json"
        try:
            with out.open("x") as f:
                f.write(text)
        except FileExistsError:
            continue
        return out
    raise FileExistsError(f"no free result file name in {out_dir}")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "scenario.toml")
    scenario = Scenario.load(path)
    data, solver = scenario.data, scenario.solver
    print(f"scenario {path}: {data}, {solver}")
    scale = solver.scale

    d = to_units(
        get_distances(data.parents, data.mu, data.sigma, (0, 1), data.seed), scale
    )
    start = solve_heuristic(d, solver.donors, seeds=solver.seeds)
    print(start.summary("heuristic", scale))
    best = solve_alns(
        d,
        start.donors,
        max_iters=solver.max_iters,
        patience=solver.patience,
        time_limit=solver.time_limit,
        runs=solver.runs,
        seed=solver.seed,
        max_sets=solver.max_sets,
    )
    print(best.summary("alns", scale))
    print(f"alns tau is {1 - best.tau / start.tau:.1%} below the heuristic")

    tau = best.tau / scale
    print(f"{len(best.sets)} donor sets at tau={tau:.4f}, fewest parents at tau first:")
    for donors, count in zip(best.sets, best.counts, strict=True):
        print(f"  {donors} ({count} parents at tau)")
        if not covers(d, donors, best.tau, solver.donors):
            raise RuntimeError(f"donor set {donors} leaves a parent uncovered")
    if not verify(d, best.donors, best.tau, solver.donors):
        raise RuntimeError("cp-sat: the best donor set leaves a parent uncovered")
    print("every set covers all parents within tau; cp-sat confirms the best")

    result: dict[str, object] = {
        "heuristic_tau": start.tau / scale,
        "tau": tau,
        "sets": [
            {"donors": donors, "parents_at_tau": count}
            for donors, count in zip(best.sets, best.counts, strict=True)
        ],
    }
    if solver.cpsat_search:
        result["cpsat_search"] = cpsat_search(
            d, solver.donors, best.tau, best.donors, solver
        )
    out = write_output(path, scenario, result)
    print(f"wrote {out}")


def cpsat_search(
    d: np.ndarray, k: int, tau: int, donors: list[int], solver: Solver
) -> dict[str, object]:
    """Run and print the CP-SAT descent from `donors` at tau; return its summary."""
    scale = solver.scale
    print(
        f"cp-sat search from tau={tau / scale:.4f}, {solver.cpsat_timeout:g}s per check:"
    )
    checks = descend(d, k, tau, solver.cpsat_timeout, donors, solver.cpsat_log)
    for c in checks:
        print(f"  tau={c.tau / scale:.4f}: {c.result} in {c.seconds:.1f}s")
    lowest: Check | None = None  # the last feasible check: the lowest tau with a cover
    for c in checks:
        if c.donors is not None:
            if not covers(d, c.donors, c.tau, k):
                raise RuntimeError(
                    f"cp-sat: cover at tau={c.tau} leaves a parent uncovered"
                )
            lowest = c
    optimal = checks[-1].result == "infeasible" and lowest is not None
    if lowest is None:
        print("  cp-sat found no cover within the time limit")
    elif optimal:
        print(
            f"  tau={lowest.tau / scale:.4f} is optimal: cp-sat proved nothing lower exists"
        )
    else:
        print(f"  lowest tau reached {lowest.tau / scale:.4f}, not proven optimal")
    return {
        "timeout": solver.cpsat_timeout,
        "checks": [
            {"tau": c.tau / scale, "result": c.result, "seconds": round(c.seconds, 3)}
            for c in checks
        ],
        "tau": None if lowest is None else lowest.tau / scale,
        "donors": None if lowest is None else lowest.donors,
        "optimal": optimal,
    }


if __name__ == "__main__":
    main()
