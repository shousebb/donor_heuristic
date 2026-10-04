import sys
from pathlib import Path

import numpy as np
import pytest
from donors import _core
from donors.donor import (
    SCALE,
    Data,
    Scenario,
    Solver,
    get_distances,
    main,
    nn_lower_bound,
    radius,
    solve_alns,
    solve_cpsat,
    solve_heuristic,
    to_bytes,
    to_units,
    verify,
)
from numpy.typing import NDArray

K = 4
Distances = NDArray[np.float64]


@pytest.fixture(scope="module")
def distances() -> Distances:
    return get_distances(n=40, mu=0.7, sigma=0.08, bounds=(0, 1), seed=1)


@pytest.fixture(scope="module")
def optimum(distances: Distances) -> float:
    """Proven optimal tau from CP-SAT with no gap."""
    start, _, _ = solve_heuristic(distances, K)
    _, tau, lower = solve_cpsat(distances, K, start=start, gap=0.0, time_limit=60)
    assert tau == lower
    return tau


# Data and helpers


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
    coarse, coarse_tau, _ = solve_heuristic(distances, K, scale=100)
    fine, fine_tau, _ = solve_heuristic(distances, K, scale=10_000)
    assert verify(distances, coarse, coarse_tau, K, scale=100)
    assert verify(distances, fine, fine_tau, K, scale=10_000)
    assert round(coarse_tau * 100) == coarse_tau * 100  # multiple of 0.01
    assert not verify(distances, fine, fine_tau - 1e-4, K, scale=10_000)


def test_radius_is_worst_nearest_donor_distance():
    d = np.array([[0, 5, 9], [5, 0, 3], [9, 3, 0]])
    assert radius(d, [0]) == 9
    assert radius(d, [1]) == 5
    assert radius(d, [0, 2]) == 3


def test_nn_lower_bound_is_valid(distances: Distances, optimum: float):
    assert nn_lower_bound(to_units(distances), K) / SCALE <= optimum


def test_to_bytes_rejects_values_beyond_u16():
    with pytest.raises(AssertionError):
        to_bytes(np.array([[0, 70_000], [70_000, 0]]))


# Rust heuristics


def test_heuristic_is_a_valid_cover(distances: Distances, optimum: float):
    selected, tau, _ = solve_heuristic(distances, K)
    assert len(set(selected)) == K
    assert verify(distances, selected, tau, K)
    assert tau >= optimum


def test_more_seeds_never_worse(distances: Distances):
    _, few, _ = solve_heuristic(distances, K, seeds=3)
    _, many, _ = solve_heuristic(distances, K, seeds=30)
    assert many <= few


def test_heuristic_is_deterministic(distances: Distances):
    assert solve_heuristic(distances, K) == solve_heuristic(distances, K)


def test_alns_reaches_optimum(distances: Distances, optimum: float):
    start, _, _ = solve_heuristic(distances, K, seeds=1)
    selected, tau, _ = solve_alns(
        distances, start, max_iters=20_000, patience=5_000, runs=2
    )
    assert verify(distances, selected, tau, K)
    assert tau == optimum


def test_optimize_reports_every_run(distances: Distances):
    d = to_units(distances)
    start = _core.construct(to_bytes(d), len(d), K, seeds=1)["donors"]
    result = _core.optimize(to_bytes(d), len(d), start, max_iters=2_000, runs=3)
    assert len(result["run_taus"]) == 3
    assert result["tau"] == min(result["run_taus"])
    assert result["tau"] <= result["init_tau"]
    assert result["tau"] == radius(d, result["donors"])
    best_taus = [best for _, _, best in result["progress"]]
    assert best_taus == sorted(best_taus, reverse=True)


def test_optimize_accepts_config(distances: Distances):
    d = to_units(distances)
    start = list(range(K))
    config = {"destroy_max": 1, "cooling": 0.99, "segment": 10}
    result = _core.optimize(to_bytes(d), len(d), start, max_iters=500, config=config)
    assert len(result["donors"]) == K


def test_optimize_stops_at_max_iters(distances: Distances):
    d = to_units(distances)
    result = _core.optimize(
        to_bytes(d), len(d), list(range(K)), max_iters=300, patience=0
    )
    assert result["run_iters"] == [300]


def test_optimize_stops_early_without_improvement(distances: Distances):
    d = to_units(distances)
    result = _core.optimize(
        to_bytes(d), len(d), list(range(K)), max_iters=1_000_000, patience=200
    )
    assert result["iters"] < 1_000_000


def test_rust_rejects_bad_input(distances: Distances):
    d = to_units(distances)
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


# CP-SAT


def test_cpsat_widens_window_to_reach_optimum(distances: Distances, optimum: float):
    start, _, _ = solve_heuristic(distances, K, seeds=1)
    selected, tau, lower = solve_cpsat(
        distances, K, start=start, gap=0.0, window=0.001, time_limit=60
    )
    assert tau == lower == optimum
    assert verify(distances, selected, tau, K)


# Z3


def test_verify_rejects_bad_covers(distances: Distances):
    selected, tau, _ = solve_heuristic(distances, K)
    assert not verify(distances, selected, tau - 1 / SCALE, K)
    assert not verify(distances, selected, tau, K - 1)


# Scenario and CLI


def test_scenario_defaults_when_tables_omitted(tmp_path: Path):
    path = tmp_path / "scenario.toml"
    path.write_text("")
    assert Scenario.load(path) == Scenario(Data(), Solver())


def test_scenario_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "scenario.toml"
    path.write_text("[solver]\nmax_iter = 10\n")
    with pytest.raises(TypeError):
        Scenario.load(path)


def test_repo_scenario_is_valid():
    """The checked-in scenario loads (no unknown keys) with usable settings."""
    scenario = Scenario.load(Path(__file__).parent.parent / "scenario.toml")
    data, solver = scenario.data, scenario.solver
    assert 0 < solver.donors < data.parents
    assert solver.seeds >= 1 and solver.runs >= 1 and solver.max_iters >= 1
    assert 1 <= solver.scale <= 65_535


def test_cli_verifies_final_cover(
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
""")
    monkeypatch.setattr(sys, "argv", ["donors", str(path)])
    main()
    out = capsys.readouterr().out
    assert "z3: all parents covered within tau" in out
