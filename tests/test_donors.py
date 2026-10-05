import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from donors import _core
from donors.cli import Data, Scenario, Solver, main, write_output
from donors.cpsat import check_tau, descend, nn_lower_bound, solve_cpsat
from donors.distances import (
    SCALE,
    get_distances,
    parents_at_tau,
    radius,
    to_bytes,
    to_units,
)
from donors.heuristics import solve_alns, solve_heuristic
from donors.verify import covers, verify
from numpy.typing import NDArray

K = 4
Distances = NDArray[np.float64]
Units = NDArray[np.int_]


@pytest.fixture(scope="module")
def distances() -> Distances:
    return get_distances(n=40, mu=0.7, sigma=0.08, bounds=(0, 1), seed=1)


@pytest.fixture(scope="module")
def d(distances: Distances) -> Units:
    return to_units(distances)


@pytest.fixture(scope="module")
def optimum(d: Units) -> int:
    """Proven optimal tau (units of 1 / SCALE) from CP-SAT with no gap."""
    start = solve_heuristic(d, K).donors
    solution = solve_cpsat(d, K, start=start, gap=0.0, time_limit=60)
    assert solution.tau == solution.lower
    return solution.tau


def assert_valid_sets(d: Units, sets: list[list[int]], counts: list[int], tau: int):
    """Distinct sorted k-sets, all at exactly tau, with correct ascending counts."""
    assert len({tuple(s) for s in sets}) == len(sets)
    for donors, count in zip(sets, counts, strict=True):
        assert donors == sorted(donors) and len(set(donors)) == K
        assert radius(d, donors) == tau
        assert parents_at_tau(d, donors) == count
    assert counts == sorted(counts)


# Distances and helpers


def test_distances_are_symmetric_with_zero_diagonal(distances: Distances):
    assert np.array_equal(distances, distances.T)
    assert np.all(np.diag(distances) == 0)
    assert distances.min() >= 0 and distances.max() <= 1


def test_distances_are_reproducible():
    a = get_distances(n=10, mu=0.7, sigma=0.08, bounds=(0, 1), seed=3)
    b = get_distances(n=10, mu=0.7, sigma=0.08, bounds=(0, 1), seed=3)
    assert np.array_equal(a, b)


def test_scale_rounds_to_integer_units():
    d = to_units(np.array([[0.0, 12.4 / SCALE], [12.6 / SCALE, 1.0]]))
    assert d.tolist() == [[0, 12], [13, SCALE]]


def test_to_units_uses_given_scale():
    d = np.array([[0.0, 0.12344], [0.12346, 1.0]])
    assert to_units(d, 10_000).tolist() == [[0, 1234], [1235, 10_000]]
    assert to_units(d, 100).tolist() == [[0, 12], [12, 100]]


def test_finer_scale_resolves_tau_more_precisely(distances: Distances):
    coarse_d, fine_d = to_units(distances, 100), to_units(distances, 10_000)
    coarse = solve_heuristic(coarse_d, K)
    fine = solve_heuristic(fine_d, K)
    assert verify(coarse_d, coarse.donors, coarse.tau, K)
    assert verify(fine_d, fine.donors, fine.tau, K)
    assert not verify(fine_d, fine.donors, fine.tau - 1, K)
    assert abs(coarse.tau / 100 - fine.tau / 10_000) <= 0.01


def test_radius_and_parents_at_tau():
    d = np.array([[0, 5, 9], [5, 0, 3], [9, 3, 0]])
    assert radius(d, [0]) == 9
    assert radius(d, [1]) == 5
    assert radius(d, [0, 2]) == 3
    assert parents_at_tau(d, [1]) == 1
    assert parents_at_tau(np.array([[0, 4], [4, 0]]), [0, 1]) == 2


def test_nn_lower_bound_is_valid(d: Units, optimum: int):
    assert nn_lower_bound(d, K) <= optimum


def test_to_bytes_rejects_values_beyond_u16():
    with pytest.raises(AssertionError):
        to_bytes(np.array([[0, 70_000], [70_000, 0]]))


# Rust heuristics


def test_heuristic_is_a_valid_cover(d: Units, optimum: int):
    solution = solve_heuristic(d, K)
    assert len(set(solution.donors)) == K
    assert verify(d, solution.donors, solution.tau, K)
    assert solution.tau >= optimum
    assert solution.lower is None
    assert solution.sets == [solution.donors]
    assert solution.counts == [parents_at_tau(d, solution.donors)]


def test_more_seeds_never_worse(d: Units):
    assert solve_heuristic(d, K, seeds=30).tau <= solve_heuristic(d, K, seeds=3).tau


def test_heuristic_is_deterministic(d: Units):
    a, b = solve_heuristic(d, K), solve_heuristic(d, K)
    assert (a.donors, a.tau, a.count) == (b.donors, b.tau, b.count)


def test_alns_reaches_optimum(d: Units, optimum: int):
    start = solve_heuristic(d, K, seeds=1).donors
    solution = solve_alns(d, start, max_iters=20_000, patience=5_000, runs=2)
    assert verify(d, solution.donors, solution.tau, K)
    assert solution.tau == optimum
    assert solution.sets == [solution.donors]


def test_alns_sets_are_distinct_covers_at_optimum(d: Units, optimum: int):
    start = solve_heuristic(d, K, seeds=1).donors
    solution = solve_alns(
        d, start, max_iters=20_000, patience=5_000, runs=2, max_sets=20
    )
    assert solution.tau == optimum
    assert 1 <= len(solution.sets) <= 20
    assert solution.sets[0] == solution.donors
    assert solution.counts[0] == solution.count
    assert_valid_sets(d, solution.sets, solution.counts, solution.tau)


def test_optimize_tied_runs_report_sets0_as_donors():
    """Runs tying on (tau, count) with different sets once reported different bests."""
    d = to_units(get_distances(200, 0.7, 0.06, (0, 1), seed=5))
    for seed in range(5):
        result = _core.optimize(
            to_bytes(d), len(d), list(range(5)), max_iters=1_500, runs=3, seed=seed,
            max_sets=5,
        )  # fmt: skip
        assert result["sets"][0] == result["donors"]
        assert result["counts"][0] == result["count"]
        assert result["tau"] == min(result["run_taus"]) == radius(d, result["donors"])
        assert result["counts"] == sorted(result["counts"])


def test_optimize_reports_every_run(d: Units):
    start = _core.construct(to_bytes(d), len(d), K, seeds=1)["donors"]
    result = _core.optimize(to_bytes(d), len(d), start, max_iters=2_000, runs=3)
    assert len(result["run_taus"]) == len(result["run_iters"]) == 3
    assert result["tau"] == min(result["run_taus"])
    assert result["tau"] <= result["init_tau"]
    assert result["tau"] == radius(d, result["donors"])
    best_taus = [best for _, _, best in result["progress"]]
    assert best_taus == sorted(best_taus, reverse=True)
    assert best_taus[-1] == result["tau"]


def test_optimize_accepts_config(d: Units):
    config = {"destroy_max": 1, "cooling": 0.99, "segment": 10}
    result = _core.optimize(
        to_bytes(d), len(d), list(range(K)), max_iters=500, config=config
    )
    assert len(result["donors"]) == K


def test_optimize_rejects_unknown_config_key(d: Units):
    with pytest.raises(ValueError, match="destory_max"):
        _core.optimize(to_bytes(d), len(d), list(range(K)), config={"destory_max": 1})


def test_optimize_stops_at_max_iters(d: Units):
    result = _core.optimize(
        to_bytes(d), len(d), list(range(K)), max_iters=300, patience=0
    )
    assert result["run_iters"] == [300]


def test_optimize_stops_early_without_improvement(d: Units):
    result = _core.optimize(
        to_bytes(d), len(d), list(range(K)), max_iters=1_000_000, patience=200
    )
    assert result["iters"] < 1_000_000


def test_rust_rejects_bad_input(d: Units):
    with pytest.raises(ValueError):
        _core.construct(to_bytes(d)[:-2], len(d), K)
    with pytest.raises(ValueError):
        _core.construct(to_bytes(d), len(d), len(d))
    with pytest.raises(ValueError):
        _core.construct(to_bytes(d), len(d), K, seeds=0)
    with pytest.raises(ValueError):
        _core.optimize(to_bytes(d), len(d), [len(d)])
    with pytest.raises(ValueError):
        _core.optimize(to_bytes(d), len(d), [0], runs=0)
    with pytest.raises(ValueError):
        _core.optimize(to_bytes(d), len(d), [0], max_sets=0)


# CP-SAT


def test_cpsat_widens_window_to_reach_optimum(d: Units, optimum: int):
    start = solve_heuristic(d, K, seeds=1).donors
    solution = solve_cpsat(d, K, start=start, gap=0.0, window=0.001, time_limit=60)
    assert solution.tau == solution.lower == optimum
    assert verify(d, solution.donors, solution.tau, K)
    assert solution.count == parents_at_tau(d, solution.donors)


# Cover checks


def test_checks_reject_bad_covers(d: Units):
    solution = solve_heuristic(d, K)
    for check in (covers, verify):
        assert check(d, solution.donors, solution.tau, K)
        assert not check(d, solution.donors, solution.tau - 1, K)
        assert not check(d, solution.donors, solution.tau, K - 1)


# Scenario, output and CLI


def test_scenario_defaults_when_tables_omitted(tmp_path: Path):
    path = tmp_path / "scenario.toml"
    path.write_text("")
    assert Scenario.load(path) == Scenario(Data(), Solver())


def test_scenario_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "scenario.toml"
    path.write_text("[solver]\nmax_iter = 10\n")
    with pytest.raises(TypeError):
        Scenario.load(path)


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ("[solver]\nmax_sets = 0", "max_sets >= 1"),
        ("[solver]\ncpsat_timeout = 0", "cpsat_timeout > 0"),
        ("[solver]\nscale = 70_000", "scale"),
        ("[solver]\nruns = 0\nseeds = 0", "seeds >= 1, runs >= 1"),
        ("[data]\nparents = 10\n[solver]\ndonors = 10", "donors < parents"),
    ],
)
def test_scenario_rejects_bad_values(tmp_path: Path, toml: str, message: str):
    path = tmp_path / "scenario.toml"
    path.write_text(toml)
    with pytest.raises(ValueError, match=message):
        Scenario.load(path)


def test_repo_scenario_is_valid():
    """The checked-in scenario loads: no unknown keys and valid values."""
    Scenario.load(Path(__file__).parent.parent / "scenario.toml")


def test_write_output_never_replaces_a_result(tmp_path: Path):
    scenario = Scenario(solver=Solver(time_limit=math.inf))
    path = tmp_path / "my.toml"
    first = write_output(path, scenario, {"tau": 1}, out_root=tmp_path)
    second = write_output(path, scenario, {"tau": 2}, out_root=tmp_path)
    assert first != second and first.parent == second.parent
    assert first.parent.name.startswith("my_")
    assert json.loads(first.read_text())["result"] == {"tau": 1}
    saved = json.loads(second.read_text())
    assert saved["result"] == {"tau": 2}
    assert saved["scenario"]["solver"]["time_limit"] is None
    assert saved["scenario"]["file"] == str(path.resolve())


def test_cli_checks_sets_and_writes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "scenario.toml"
    path.write_text("""
[data]
parents = 30

[solver]
donors = 3
runs = 2
max_iters = 2_000
patience = 500
scale = 10_000
max_sets = 5
cpsat_search = true
""")
    monkeypatch.setattr(sys, "argv", ["donors", str(path)])
    monkeypatch.chdir(tmp_path)
    main()
    out = capsys.readouterr().out
    assert "every set covers all parents within tau; cp-sat confirms the best" in out

    (result_file,) = (tmp_path / "outputs").glob("scenario_*/result_*.json")
    saved = json.loads(result_file.read_text())
    assert saved["scenario"]["data"]["parents"] == 30
    result = saved["result"]
    assert result["tau"] <= result["heuristic_tau"]
    sets = result["sets"]
    assert 1 <= len(sets) <= 5
    counts = [s["parents_at_tau"] for s in sets]
    assert counts == sorted(counts) and counts[0] >= 1

    search = result["cpsat_search"]
    assert search["optimal"] and search["checks"][-1]["result"] == "infeasible"
    assert search["tau"] <= result["tau"]
    assert "is optimal: cp-sat proved nothing lower exists" in out


# CP-SAT search


def test_descend_proves_optimum_from_heuristic(d: Units, optimum: int):
    start = solve_heuristic(d, K, seeds=1)
    checks = descend(d, K, start.tau)
    assert [c.tau for c in checks] == list(range(start.tau, optimum - 2, -1))
    feasible = ["feasible"] * (start.tau - optimum + 1)
    assert [c.result for c in checks] == [*feasible, "infeasible"]
    for c in checks[:-1]:
        assert c.donors is not None and covers(d, c.donors, c.tau, K)
    assert checks[-1].donors is None
    assert all(c.seconds >= 0 for c in checks)


def test_descend_confirms_given_donors_first(d: Units, optimum: int):
    start = solve_heuristic(d, K, seeds=1)
    checks = descend(d, K, start.tau, donors=start.donors)
    assert checks[0].donors == start.donors
    assert checks[-1].result == "infeasible" and checks[-1].tau == optimum - 1


def test_check_tau_with_fixed_donors_rejects_a_non_cover(d: Units):
    start = solve_heuristic(d, K)
    check = check_tau(d, K, start.tau - 1, timeout=10, donors=start.donors)
    assert check.result == "infeasible"


def test_check_tau_gives_up_after_timeout():
    d = to_units(get_distances(400, 0.7, 0.06, (0, 1), seed=2), 1000)
    start = solve_heuristic(d, 10, seeds=1)
    check = check_tau(d, 10, start.tau - 30, timeout=0.01)
    assert check.result == "unknown" and check.donors is None


def test_check_tau_logs_search_progress_only_when_asked(
    d: Units, capfd: pytest.CaptureFixture[str]
):
    start = solve_heuristic(d, K)
    check_tau(d, K, start.tau, timeout=10)
    assert "CP-SAT solver" not in capfd.readouterr().out
    check_tau(d, K, start.tau, timeout=10, log=True)
    assert "CP-SAT solver" in capfd.readouterr().out
