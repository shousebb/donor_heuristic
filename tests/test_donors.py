import sys

import numpy as np
import pytest
from donors import _core
from donors.donor import (
    SCALE,
    get_distances,
    main,
    nn_lower_bound,
    radius,
    scale,
    solve_alns,
    solve_cpsat,
    solve_heuristic,
    to_bytes,
    verify,
    verify_lower,
    z3_cover_within,
    z3_lower_bound,
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
    d = scale(np.array([[0.0, 12.4 / SCALE], [12.6 / SCALE, 1.0]]))
    assert d.tolist() == [[0, 12], [13, SCALE]]


def test_radius_is_worst_nearest_donor_distance():
    d = np.array([[0, 5, 9], [5, 0, 3], [9, 3, 0]])
    assert radius(d, [0]) == 9
    assert radius(d, [1]) == 5
    assert radius(d, [0, 2]) == 3


def test_nn_lower_bound_is_valid(distances: Distances, optimum: float):
    assert nn_lower_bound(scale(distances), K) / SCALE <= optimum


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
    selected, tau, _ = solve_alns(distances, start, time_limit=1, runs=2)
    assert verify(distances, selected, tau, K)
    assert tau == optimum


def test_optimize_reports_every_run(distances: Distances):
    d = scale(distances)
    start = _core.construct(to_bytes(d), len(d), K, seeds=1)["donors"]
    result = _core.optimize(to_bytes(d), len(d), start, 0.2, runs=3)
    assert len(result["run_taus"]) == 3
    assert result["tau"] == min(result["run_taus"])
    assert result["tau"] <= result["init_tau"]
    assert result["tau"] == radius(d, result["donors"])
    best_taus = [best for _, _, best in result["progress"]]
    assert best_taus == sorted(best_taus, reverse=True)


def test_optimize_accepts_config(distances: Distances):
    d = scale(distances)
    start = list(range(K))
    config = {"destroy_max": 1, "cooling": 0.99, "segment": 10}
    result = _core.optimize(to_bytes(d), len(d), start, 0.2, config=config)
    assert len(result["donors"]) == K


def test_rust_rejects_bad_input(distances: Distances):
    d = scale(distances)
    with pytest.raises(ValueError):
        _core.construct(to_bytes(d)[:-2], len(d), K)
    with pytest.raises(ValueError):
        _core.construct(to_bytes(d), len(d), len(d))
    with pytest.raises(ValueError):
        _core.construct(to_bytes(d), len(d), K, seeds=0)
    with pytest.raises(ValueError):
        _core.optimize(to_bytes(d), len(d), [len(d)], 1.0)
    with pytest.raises(ValueError):
        _core.optimize(to_bytes(d), len(d), [0], 1.0, runs=0)


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


def test_z3_lower_bound(distances: Distances, optimum: float):
    assert verify_lower(distances, optimum, K, time_limit=60) == "proven"
    assert verify_lower(distances, optimum + 1 / SCALE, K, time_limit=60) == "refuted"


def test_z3_cover_within_returns_a_valid_cover(distances: Distances, optimum: float):
    d = scale(distances)
    status, donors = z3_cover_within(d, round(optimum * SCALE), K, time_limit=60)
    assert status == "refuted" and donors is not None
    assert len(donors) <= K
    assert radius(d, donors) <= round(optimum * SCALE)


def test_z3_binary_search_reaches_optimum(distances: Distances, optimum: float):
    _, upper, _ = solve_heuristic(distances, K, seeds=1)
    lower, better = z3_lower_bound(distances, K, 0.0, upper, time_limit=120)
    assert lower == optimum
    if better is not None:
        assert verify(distances, better, upper, K)


# CLI


@pytest.mark.parametrize("cpsat_time", ["0", "10"])
def test_cli_reports_bounds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cpsat_time: str
):
    argv = ["donors", "--parents", "30", "--donors", "3", "--alns-time", "0.2"]
    argv += ["--runs", "2", "--cpsat-time", cpsat_time, "--z3-time", "30"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    out = capsys.readouterr().out
    assert "z3: all parents covered within tau" in out
    assert "optimal tau in [" in out
