// Donor-selection search (discrete k-center), independent of Python.
//
// Choose k donors from n recurrent parents to minimize tau, the largest distance from a
// parent to its nearest donor; ties are broken by fewer parents at tau. Distances are
// integers (units of 1 / scale, see donors.distances).
//
// construct_best: multi-start greedy construction + swap descent.
// alns: one ALNS run. Each iteration destroys a few donors (random, nearest the
// bottleneck parent, or mutually related), repairs greedily, then runs swap descent.
// Simulated annealing decides acceptance; operator weights adapt to their success, as in
// alns_rs. Each run keeps a Pool of the best distinct donor sets at its lowest tau.

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::time::Instant;

use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;

/// Adaptive-search tuning; overridable per call via the optional `config` dict.
pub struct Config {
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
    /// Defaults overridden by `pairs`; an unknown key is an error naming it.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, f64)>) -> Result<Self, String> {
        let mut c = Config::default();
        for (key, v) in pairs {
            match key.as_str() {
                "segment" => c.segment = v as u64,
                "react" => c.react = v,
                "score_best" => c.score_best = v,
                "score_better" => c.score_better = v,
                "score_accept" => c.score_accept = v,
                "cooling" => c.cooling = v,
                "temp_factor" => c.temp_factor = v,
                "destroy_max" => c.destroy_max = v as usize,
                _ => return Err(format!("unknown config key {key:?}")),
            }
        }
        Ok(c)
    }
}

/// (tau, parents at tau): lexicographic, lower is better.
pub type Score = (u16, u32);
const WORST: Score = (u16::MAX, u32::MAX);

pub struct Dist {
    pub n: usize,
    pub d: Vec<u16>,
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

/// The best distinct donor sets at the lowest tau seen so far: at most `cap`
/// (score, sorted donors) pairs, ordered by score, then donors.
pub struct Pool {
    pub tau: u16,
    cap: usize,
    pub sets: BTreeSet<(Score, Vec<usize>)>,
}

impl Pool {
    pub fn new(cap: usize) -> Self {
        Pool {
            tau: u16::MAX,
            cap,
            sets: BTreeSet::new(),
        }
    }

    /// Offer `sel` with its score. A lower tau empties the pool; a full pool drops its
    /// worst set.
    pub fn record(&mut self, sel: &[usize], s: Score) {
        if s.0 < self.tau {
            self.tau = s.0;
            self.sets.clear();
        }
        let full = self.sets.len() >= self.cap;
        if s.0 > self.tau || full && self.sets.last().is_some_and(|(worst, _)| s > *worst) {
            return;
        }
        let mut set = sel.to_vec();
        set.sort_unstable();
        self.sets.insert((s, set));
        if self.sets.len() > self.cap {
            self.sets.pop_last();
        }
    }
}

pub struct Outcome {
    pub pool: Pool,
    pub init_tau: u16,
    pub iters: u64,
    pub wall: f64,
    pub progress: Vec<(u64, u16, u16)>,
}

/// When an ALNS run stops: whichever limit is reached first.
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_iters: u64, // iteration cap
    pub patience: u64,  // stop after this many iterations without a new best (0 = off)
    pub time_s: f64,    // wall-clock cap in seconds
}

/// One ALNS run from `init` (k = len(init)), keeping up to `max_sets` sets at its best tau.
pub fn alns(
    dist: &Dist,
    init: Vec<usize>,
    limits: Limits,
    seed: u64,
    cfg: &Config,
    max_sets: usize,
) -> Outcome {
    let k = init.len();
    let mut rng = XorShiftRng::seed_from_u64(seed);
    // Energy for annealing: tau plus count as a fraction, preserving the lexicographic order
    let energy = |s: Score| s.0 as f64 + s.1 as f64 / (dist.n as f64 + 1.0);

    let mut cur_sel = init;
    let mut cur = swap_descent(dist, &mut cur_sel);
    let init_tau = cur.0;
    let mut best = cur;
    let mut pool = Pool::new(max_sets);
    pool.record(&cur_sel, cur);

    let mut temp = (cur.0 as f64 * cfg.temp_factor).max(1.0);
    let mut weights = [1.0f64; 3];
    let mut scores = [0.0f64; 3];
    let mut uses = [0u64; 3];
    let max_q = cfg.destroy_max.clamp(1, k.saturating_sub(1).max(1));

    let start = Instant::now();
    let mut progress = Vec::new();
    let mut last_emit = 0.0;
    let mut it = 0u64;
    let mut since_best = 0u64;
    while it < limits.max_iters && start.elapsed().as_secs_f64() < limits.time_s {
        if limits.patience > 0 && since_best >= limits.patience {
            break;
        }
        it += 1;
        since_best += 1;
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
        pool.record(&sel, new);

        let delta = energy(new) - energy(cur);
        if delta <= 0.0 || rand_f64(&mut rng) < (-delta / temp).exp() {
            if new < best {
                best = new;
                since_best = 0;
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
    Outcome {
        pool,
        init_tau,
        iters: it,
        wall: start.elapsed().as_secs_f64(),
        progress,
    }
}

/// Best of `seeds` greedy + swap descent runs. Seed 0 starts empty; the others start
/// from distinct random first donors (a prefix of one random order, so more seeds
/// always include the fewer-seed runs).
pub fn construct_best(dist: &Dist, k: usize, seeds: usize, seed: u64) -> (Vec<usize>, Score) {
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

/// The best `max_sets` distinct (score, donors) pairs at the lowest tau over all runs,
/// best first.
pub fn merge_sets(outcomes: &[Outcome], max_sets: usize) -> Vec<(Score, Vec<usize>)> {
    let tau = outcomes.iter().map(|o| o.pool.tau).min().unwrap();
    let union: BTreeSet<&(Score, Vec<usize>)> = outcomes
        .iter()
        .filter(|o| o.pool.tau == tau)
        .flat_map(|o| &o.pool.sets)
        .collect();
    union.into_iter().take(max_sets).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symmetric n x n distances in 0..100 with a zero diagonal, from a fixed seed.
    fn random_dist(n: usize, seed: u64) -> Dist {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let mut d = vec![0u16; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let v = (rng.next_u64() % 100) as u16;
                (d[i * n + j], d[j * n + i]) = (v, v);
            }
        }
        Dist { n, d }
    }

    fn sets(pool: &Pool) -> Vec<(Score, Vec<usize>)> {
        pool.sets.iter().cloned().collect()
    }

    #[test]
    fn pool_keeps_the_best_sets_at_the_lowest_tau() {
        let mut pool = Pool::new(2);
        pool.record(&[3, 1], (9, 1));
        pool.record(&[2, 0], (8, 5));
        assert_eq!(sets(&pool), [((8, 5), vec![0, 2])]); // lower tau empties the pool
        pool.record(&[4, 5], (8, 3));
        pool.record(&[6, 7], (8, 4));
        assert_eq!(sets(&pool), [((8, 3), vec![4, 5]), ((8, 4), vec![6, 7])]);
        pool.record(&[1, 9], (8, 1));
        pool.record(&[5, 4], (8, 3)); // duplicate in another order
        pool.record(&[8, 9], (8, 9)); // worse than every kept set
        pool.record(&[2, 3], (9, 0)); // higher tau
        assert_eq!(sets(&pool), [((8, 1), vec![1, 9]), ((8, 3), vec![4, 5])]);
    }

    #[test]
    fn merge_takes_the_lowest_tau_best_first() {
        let outcome = |records: &[(&[usize], Score)]| {
            let mut pool = Pool::new(3);
            for &(sel, s) in records {
                pool.record(sel, s);
            }
            Outcome {
                pool,
                init_tau: 0,
                iters: 0,
                wall: 0.0,
                progress: vec![],
            }
        };
        let outcomes = [
            outcome(&[(&[5, 6], (7, 2)), (&[1, 2], (7, 2))]),
            outcome(&[(&[0, 1], (9, 1))]),
            outcome(&[(&[1, 2], (7, 2)), (&[3, 4], (7, 1))]),
        ];
        let merged = merge_sets(&outcomes, 2);
        assert_eq!(merged, [((7, 1), vec![3, 4]), ((7, 2), vec![1, 2])]);
    }

    #[test]
    fn config_rejects_unknown_keys() {
        let ok = Config::from_pairs([("destroy_max".to_string(), 2.0)]).unwrap();
        assert_eq!(ok.destroy_max, 2);
        let err = Config::from_pairs([("destory_max".to_string(), 2.0)]).err();
        assert_eq!(err.as_deref(), Some("unknown config key \"destory_max\""));
    }

    #[test]
    fn alns_pool_scores_match_their_sets() {
        let dist = random_dist(30, 1);
        let limits = Limits {
            max_iters: 500,
            patience: 0,
            time_s: f64::INFINITY,
        };
        let out = alns(&dist, vec![0, 1, 2], limits, 0, &Config::default(), 5);
        let (best, _) = out.pool.sets.first().unwrap();
        assert_eq!(out.pool.tau, best.0);
        assert_eq!(out.progress.last().unwrap().2, best.0);
        for (s, set) in &out.pool.sets {
            assert_eq!(set.len(), 3);
            assert!(set.is_sorted());
            assert_eq!(*s, score(&dist.nearest(set)));
        }
    }
}
