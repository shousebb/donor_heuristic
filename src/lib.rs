// Python bindings (donors._core) for the donor-selection search in search.rs.
//
// Distances are passed from Python as a little-endian u16 byte buffer of the symmetric
// n x n matrix of integer distances (units of 1 / scale, see donors.distances).

mod search;

use std::collections::HashMap;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use search::{alns, construct_best, merge_sets, Config, Dist, Limits, Outcome};

fn parse_dist(dist: &[u8], n: usize) -> PyResult<Dist> {
    if dist.len() != n * n * 2 {
        return Err(PyValueError::new_err(
            "dist must hold n * n little-endian u16 values",
        ));
    }
    Ok(Dist {
        n,
        d: dist
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect(),
    })
}

/// Greedy + swap descent from `seeds` starts (see construct_best); k donors.
///
/// `dist` is the n x n distance matrix as little-endian u16 bytes. Returns a dict with
/// donors, tau, count (parents at tau) and wall.
#[pyfunction]
#[pyo3(signature = (dist, n, k, seeds=10, seed=0))]
fn construct(
    py: Python<'_>,
    dist: &[u8],
    n: usize,
    k: usize,
    seeds: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let dist = parse_dist(dist, n)?;
    if k == 0 || k >= n || seeds == 0 {
        return Err(PyValueError::new_err("need 1 <= k < n and seeds >= 1"));
    }
    let start = Instant::now();
    let (donors, score) = py.allow_threads(|| construct_best(&dist, k, seeds, seed));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("donors", donors)?;
    dict.set_item("tau", score.0)?;
    dict.set_item("count", score.1)?;
    dict.set_item("wall", start.elapsed().as_secs_f64())?;
    Ok(dict.into())
}

/// `runs` parallel ALNS searches (seeds seed..seed + runs) from the initial donors
/// `init` (k = len(init)). Each run stops after `max_iters` iterations, `patience`
/// iterations without a new best (0 = off), or `time_limit_s` seconds, whichever comes
/// first, and keeps its best `max_sets` distinct donor sets at its lowest tau.
///
/// `dist` is the n x n distance matrix as little-endian u16 bytes. Returns a dict with
/// sets and counts (the best `max_sets` sets at the lowest tau over all runs, best
/// first, with their parents at tau), donors, tau and count (of sets[0]), init_tau, wall
/// and progress [(iteration, current tau, best tau)] of the run that found sets[0],
/// iters (all runs), run_taus and run_iters.
#[pyfunction]
#[pyo3(signature = (
    dist, n, init, max_iters=1_000_000, patience=100_000, time_limit_s=f64::INFINITY,
    seed=0, runs=1, max_sets=1, config=None,
))]
#[allow(clippy::too_many_arguments)]
fn optimize(
    py: Python<'_>,
    dist: &[u8],
    n: usize,
    init: Vec<usize>,
    max_iters: u64,
    patience: u64,
    time_limit_s: f64,
    seed: u64,
    runs: u64,
    max_sets: usize,
    config: Option<HashMap<String, f64>>,
) -> PyResult<PyObject> {
    let limits = Limits {
        max_iters,
        patience,
        time_s: time_limit_s,
    };
    let dist = parse_dist(dist, n)?;
    if init.is_empty()
        || init.len() >= n
        || init.iter().any(|&i| i >= n)
        || runs == 0
        || max_sets == 0
    {
        return Err(PyValueError::new_err(
            "init must hold 1..n-1 donor indices below n, runs >= 1 and max_sets >= 1",
        ));
    }
    let cfg = Config::from_pairs(config.unwrap_or_default()).map_err(PyValueError::new_err)?;
    let outcomes: Vec<Outcome> = py.allow_threads(|| {
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..runs)
                .map(|r| {
                    let (dist, cfg, init) = (&dist, &cfg, init.clone());
                    s.spawn(move || alns(dist, init, limits, seed + r, cfg, max_sets))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    });
    let sets = merge_sets(&outcomes, max_sets);
    let best = &sets[0];
    let run = outcomes
        .iter()
        .find(|o| o.pool.sets.contains(best))
        .unwrap();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("donors", &best.1)?;
    dict.set_item("tau", best.0 .0)?;
    dict.set_item("count", best.0 .1)?;
    dict.set_item("sets", sets.iter().map(|(_, s)| s).collect::<Vec<_>>())?;
    dict.set_item("counts", sets.iter().map(|(s, _)| s.1).collect::<Vec<_>>())?;
    dict.set_item("init_tau", run.init_tau)?;
    dict.set_item("wall", run.wall)?;
    dict.set_item("progress", &run.progress)?;
    dict.set_item("iters", outcomes.iter().map(|o| o.iters).sum::<u64>())?;
    dict.set_item(
        "run_taus",
        outcomes.iter().map(|o| o.pool.tau).collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "run_iters",
        outcomes.iter().map(|o| o.iters).collect::<Vec<_>>(),
    )?;
    Ok(dict.into())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(construct, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    Ok(())
}
