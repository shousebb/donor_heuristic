// Heuristics for donor selection (discrete k-center).
//
// Choose k donors from n recurrent parents to minimize tau, the largest distance from a
// parent to its nearest donor; ties are broken by fewer parents at tau. Distances are
// integers (units of 1 / SCALE in donor.py) passed from Python as a little-endian u16
// byte buffer of the symmetric n x n matrix.
//
// construct: multi-start greedy construction + swap descent.
// optimize: parallel ALNS runs. Each iteration destroys a few donors (random, nearest the
// bottleneck parent, or mutually related), repairs greedily, then runs swap descent.
// Simulated annealing decides acceptance; operator weights adapt to their success, as in
// alns_rs.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;

// Adaptive-search tuning; overridable per call via the optional `config` dict.
struct Config {
    segment: u64,       // iterations per adaptive-weight update window
    react: f64,         // reaction factor blending new operator scores into weights
    score_best: f64,    // reward: new global best
    score_better: f64,  // reward: improved current
    score_accept: f64,  // reward: accepted-worse (SA)
    cooling: f64,       // temperature multiplier per iteration
    temp_factor: f64,   // initial temperature = initial tau * temp_factor (>= 1.0)
    destroy_max: usize, // donors destroyed per iteration: uniform in 1..=destroy_max
}

impl Default for Config {
    fn default() -> Self {
        Config {
            segment: 50,
            react: 0.2,
            score_best: 5.0,
            score_better: 3.0,
            score_accept: 1.0,
            cooling: 0.9995,
            temp_factor: 0.01,
            destroy_max: 3,
        }
    }
}

impl Config {
    fn from_map(map: Option<HashMap<String, f64>>) -> Self {
        let mut c = Config::default();
        let Some(m) = map else { return c };
        let get = |key: &str| m.get(key).copied();
        if let Some(v) = get("segment") {
            c.segment = v as u64;
        }
        if let Some(v) = get("react") {
            c.react = v;
        }
        if let Some(v) = get("score_best") {
            c.score_best = v;
        }
        if let Some(v) = get("score_better") {
            c.score_better = v;
        }
        if let Some(v) = get("score_accept") {
            c.score_accept = v;
        }
        if let Some(v) = get("cooling") {
            c.cooling = v;
        }
        if let Some(v) = get("temp_factor") {
            c.temp_factor = v;
        }
        if let Some(v) = get("destroy_max") {
            c.destroy_max = v as usize;
        }
        c
    }
}

/// (tau, parents at tau): lexicographic, lower is better.
type Score = (u16, u32);
const WORST: Score = (u16::MAX, u32::MAX);

struct Dist {
    n: usize,
    d: Vec<u16>,
}

impl Dist {
    /// Distances from donor `c` to every parent (row c; the matrix is symmetric).
    fn row(&self, c: usize) -> &[u16] {
        &self.d[c * self.n..(c + 1) * self.n]
    }

    /// Each parent's distance to its nearest donor in `donors` (u16::MAX if none).
    fn nearest(&self, donors: &[usize]) -> Vec<u16> {
        let mut r = vec![u16::MAX; self.n];
        for &s in donors {
            for (rj, &v) in r.iter_mut().zip(self.row(s)) {
                *rj = (*rj).min(v);
            }
        }
        r
    }
}

fn score(r: &[u16]) -> Score {
    let tau = *r.iter().max().unwrap();
    (tau, r.iter().filter(|&&v| v == tau).count() as u32)
}

/// Donor (not in `exclude`) whose addition to nearest-distances `r` gives the lowest
/// score strictly below `bound`, with that score.
///
/// Parents are scanned from farthest to nearest. A parent with r[j] below the running tau
/// cannot raise tau or join the count, so the scan stops there; a candidate is dropped
/// as soon as it cannot beat the best so far. Usually only a few parents are scanned.
fn best_candidate(
    dist: &Dist,
    r: &[u16],
    exclude: &[usize],
    bound: Score,
) -> Option<(usize, Score)> {
    let mut order: Vec<usize> = (0..dist.n).collect();
    order.sort_unstable_by_key(|&j| Reverse(r[j]));

    let mut best = bound;
    let mut best_c = None;
    'candidates: for c in 0..dist.n {
        if exclude.contains(&c) {
            continue;
        }
        let row = dist.row(c);
        let (mut tau, mut count) = (0u16, 0u32);
        for &j in &order {
            if r[j] < tau {
                break;
            }
            let v = r[j].min(row[j]);
            if v > tau {
                (tau, count) = (v, 1);
            } else if v == tau {
                count += 1;
            } else {
                continue;
            }
            if (tau, count) >= best {
                continue 'candidates; // tau and count only grow from here
            }
        }
        best = (tau, count);
        best_c = Some(c);
    }
    best_c.map(|c| (c, best))
}

/// Replace any donor with any parent while that strictly lowers the score.
fn swap_descent(dist: &Dist, sel: &mut [usize]) -> Score {
    let mut cur = score(&dist.nearest(sel));
    loop {
        let mut improved = false;
        for pos in 0..sel.len() {
            let others: Vec<usize> = (0..sel.len())
                .filter(|&i| i != pos)
                .map(|i| sel[i])
                .collect();
            if let Some((c, s)) = best_candidate(dist, &dist.nearest(&others), sel, cur) {
                sel[pos] = c;
                cur = s;
                improved = true;
            }
        }
        if !improved {
            return cur;
        }
    }
}

/// Greedily add donors up to k, then swap descent.
fn repair(dist: &Dist, sel: &mut Vec<usize>, k: usize) -> Score {
    while sel.len() < k {
        let (c, _) = best_candidate(dist, &dist.nearest(sel), sel, WORST).expect("n > k");
        sel.push(c);
    }
    swap_descent(dist, sel)
}

#[inline]
fn rand_f64(rng: &mut XorShiftRng) -> f64 {
    ((rng.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
}

#[inline]
fn rand_below(rng: &mut XorShiftRng, n: usize) -> usize {
    (rng.next_u64() % (n as u64)) as usize
}

/// Remove `q` donors chosen by operator `op`.
fn destroy(op: usize, dist: &Dist, sel: &mut Vec<usize>, rng: &mut XorShiftRng, q: usize) {
    match op {
        // random donors
        0 => {
            for i in 0..sel.len() {
                let j = i + rand_below(rng, sel.len() - i);
                sel.swap(i, j);
            }
        }
        // donors nearest the bottleneck parent (the one at tau)
        1 => {
            let r = dist.nearest(sel);
            let worst = (0..dist.n).max_by_key(|&j| r[j]).unwrap();
            sel.sort_by_key(|&s| dist.row(s)[worst]);
        }
        // a random donor and the donors nearest it
        _ => {
            let anchor = sel[rand_below(rng, sel.len())];
            sel.sort_by_key(|&s| dist.row(anchor)[s]);
        }
    }
    sel.drain(..q);
}

struct Outcome {
    donors: Vec<usize>,
    score: Score,
    init_tau: u16,
    iters: u64,
    wall: f64,
    progress: Vec<(u64, u16, u16)>,
}

fn alns(dist: &Dist, init: Vec<usize>, time_limit_s: f64, seed: u64, cfg: &Config) -> Outcome {
    let k = init.len();
    let mut rng = XorShiftRng::seed_from_u64(seed);
    // Energy for annealing: tau plus count as a fraction, preserving the lexicographic order
    let energy = |s: Score| s.0 as f64 + s.1 as f64 / (dist.n as f64 + 1.0);

    let mut cur_sel = init;
    let mut cur = swap_descent(dist, &mut cur_sel);
    let init_tau = cur.0;
    let (mut best_sel, mut best) = (cur_sel.clone(), cur);

    let mut temp = (cur.0 as f64 * cfg.temp_factor).max(1.0);
    let mut weights = [1.0f64; 3];
    let mut scores = [0.0f64; 3];
    let mut uses = [0u64; 3];
    let max_q = cfg.destroy_max.clamp(1, k.saturating_sub(1).max(1));

    let start = Instant::now();
    let mut progress = Vec::new();
    let mut last_emit = 0.0;
    let mut it = 0u64;
    while start.elapsed().as_secs_f64() < time_limit_s {
        it += 1;
        let q = 1 + rand_below(&mut rng, max_q);

        let total: f64 = weights.iter().sum();
        let mut pick = rand_f64(&mut rng) * total;
        let op = weights.iter().position(|&w| {
            pick -= w;
            pick <= 0.0
        });
        let op = op.unwrap_or(2);
        uses[op] += 1;

        let mut sel = cur_sel.clone();
        destroy(op, dist, &mut sel, &mut rng, q);
        let new = repair(dist, &mut sel, k);

        let delta = energy(new) - energy(cur);
        if delta <= 0.0 || rand_f64(&mut rng) < (-delta / temp).exp() {
            if new < best {
                (best_sel, best) = (sel.clone(), new);
                scores[op] += cfg.score_best;
            } else if new < cur {
                scores[op] += cfg.score_better;
            } else {
                scores[op] += cfg.score_accept;
            }
            (cur_sel, cur) = (sel, new);
        }
        temp = (temp * cfg.cooling).max(1e-6);

        if it.is_multiple_of(cfg.segment) {
            for i in 0..3 {
                if uses[i] > 0 {
                    weights[i] =
                        (1.0 - cfg.react) * weights[i] + cfg.react * (scores[i] / uses[i] as f64);
                }
                (scores[i], uses[i]) = (0.0, 0);
            }
        }

        let now = start.elapsed().as_secs_f64();
        if now - last_emit >= 0.5 {
            last_emit = now;
            progress.push((it, cur.0, best.0));
        }
    }
    progress.push((it, cur.0, best.0));
    best_sel.sort_unstable();
    Outcome {
        donors: best_sel,
        score: best,
        init_tau,
        iters: it,
        wall: start.elapsed().as_secs_f64(),
        progress,
    }
}

/// Best of `seeds` greedy + swap descent runs. Seed 0 starts empty; the others start
/// from distinct random first donors (a prefix of one random order, so more seeds
/// always include the fewer-seed runs).
fn construct_best(dist: &Dist, k: usize, seeds: usize, seed: u64) -> (Vec<usize>, Score) {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..dist.n).collect();
    for i in 0..seeds.saturating_sub(1).min(dist.n) {
        let j = i + rand_below(&mut rng, dist.n - i);
        order.swap(i, j);
    }
    let firsts = std::iter::once(None).chain(
        order[..seeds.saturating_sub(1).min(dist.n)]
            .iter()
            .map(|&f| Some(f)),
    );
    firsts
        .map(|first| {
            let mut sel: Vec<usize> = first.into_iter().collect();
            let s = repair(dist, &mut sel, k);
            sel.sort_unstable();
            (sel, s)
        })
        .min_by_key(|(_, s)| *s)
        .unwrap()
}

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
/// `init` (k = len(init)), each for `time_limit_s` seconds; returns the best.
///
/// `dist` is the n x n distance matrix as little-endian u16 bytes. Returns a dict with
/// donors, tau, count (parents at tau), init_tau, iters (all runs), wall, run_taus and
/// progress [(iteration, current tau, best tau)] of the best run.
#[pyfunction]
#[pyo3(signature = (dist, n, init, time_limit_s, seed=0, runs=1, config=None))]
#[allow(clippy::too_many_arguments)]
fn optimize(
    py: Python<'_>,
    dist: &[u8],
    n: usize,
    init: Vec<usize>,
    time_limit_s: f64,
    seed: u64,
    runs: u64,
    config: Option<HashMap<String, f64>>,
) -> PyResult<PyObject> {
    let dist = parse_dist(dist, n)?;
    if init.is_empty() || init.len() >= n || init.iter().any(|&i| i >= n) || runs == 0 {
        return Err(PyValueError::new_err(
            "init must hold 1..n-1 donor indices below n, and runs >= 1",
        ));
    }
    let cfg = Config::from_map(config);
    let outcomes: Vec<Outcome> = py.allow_threads(|| {
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..runs)
                .map(|r| {
                    let (dist, cfg, init) = (&dist, &cfg, init.clone());
                    s.spawn(move || alns(dist, init, time_limit_s, seed + r, cfg))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    });
    let run_taus: Vec<u16> = outcomes.iter().map(|o| o.score.0).collect();
    let iters: u64 = outcomes.iter().map(|o| o.iters).sum();
    let best = outcomes.into_iter().min_by_key(|o| o.score).unwrap();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("donors", best.donors)?;
    dict.set_item("tau", best.score.0)?;
    dict.set_item("count", best.score.1)?;
    dict.set_item("init_tau", best.init_tau)?;
    dict.set_item("iters", iters)?;
    dict.set_item("wall", best.wall)?;
    dict.set_item("run_taus", run_taus)?;
    dict.set_item("progress", best.progress)?;
    Ok(dict.into())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(construct, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    Ok(())
}
