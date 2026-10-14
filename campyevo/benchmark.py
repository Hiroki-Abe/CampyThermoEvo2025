"""Simulation matched to the directed-evolution experiment of Berdejo et al. (2024).

In that experiment, Salmonella Typhimurium LT2 was heated at 55 degC for 20 min
(a 4.22-log reduction of the parental strain), the WHOLE heated sample was
diluted into fresh broth, the survivors grew to stationary phase, and the cycle
was repeated ten times. After the last cycle one clone per lineage was isolated
and heated once more with the same treatment, next to the parental strain.

This module runs the same design with the present model:

* every cycle heats ``n_heated`` cells (default 1e8: 50 uL of a stationary
  culture at an assumed 2e9 CFU/mL);
* ALL survivors are carried over and regrow back to ``n_heated`` cells, each
  survivor contributing the same number of daughter cells;
* after the last cycle, one surviving cell is picked at random (the "clone")
  and its log reduction under the same treatment is compared with that of the
  starting strain.

To keep memory bounded, at most ``max_parents`` survivors are kept as parents
of the next cycle (a uniform random sample of all survivors, drawn by exact
reservoir sampling). Cells are processed in blocks of ``block`` cells.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .posterior import Posterior
from .sim import death_times, log_reduction, _quad


@dataclass(frozen=True)
class WholePopSettings:
    t_heating: float                 # heating time per cycle (min)
    n_heated: int = 10 ** 8          # cells heated in every cycle
    n_cycles: int = 10
    init_logpara: tuple = None       # (ln delta, ln p) of the starting strain; None = draw from MVN_inter
    max_parents: int = 100_000       # survivors kept as parents (uniform sample of all survivors)
    block: int = 2_000_000           # cells processed at once


def _reservoir_update(pool, seen, new, rng, cap):
    """Uniform sample (size <= cap) of all survivors seen so far, updated with `new`."""
    s = len(new)
    if s == 0:
        return pool, seen
    if seen + s <= cap:
        pool = new if pool is None else np.vstack([pool, new])
        return pool, seen + s
    # take x from the new batch and cap - x from the pool, x ~ hypergeometric
    x = rng.hypergeometric(s, seen, cap) if seen > 0 else cap
    keep_old = pool[rng.choice(len(pool), size=min(cap - x, len(pool)), replace=False)] if pool is not None and cap - x > 0 else np.empty((0, 2))
    take_new = new[rng.choice(s, size=x, replace=False)]
    return np.vstack([keep_old, take_new]), seen + s


def run_whole_population_chain(chain_id: int, post: dict, st: WholePopSettings, base_seed: int = 2026):
    """One lineage. Returns dict with start, clone (ln delta, ln p), and log10 survival per cycle."""
    rng = np.random.default_rng([int(base_seed), 7919, int(chain_id)])
    n_post = post["mu_inter"].shape[0]
    pid = int(rng.integers(n_post))
    mu, cov_b, cov_w = post["mu_inter"][pid], post["cov_inter"][pid], post["cov_intra"][pid]
    chol_w = np.linalg.cholesky(cov_w)
    prec_b = np.linalg.inv(cov_b)
    start = (np.asarray(st.init_logpara, float) if st.init_logpara is not None
             else rng.multivariate_normal(mu, cov_b))
    parents = start[None, :]
    log_surv = np.full(st.n_cycles, np.nan)
    th = float(st.t_heating)

    for cyc in range(st.n_cycles):
        k = len(parents)
        per = max(st.n_heated // k, 1)
        total = per * k
        q_par = _quad(parents, mu, prec_b)
        pool, seen = None, 0
        # offspring are generated parent by parent, in blocks of <= st.block cells
        idx_all_blocks = range(0, total, st.block)
        for b0 in idx_all_blocks:
            m = min(st.block, total - b0)
            pidx = (np.arange(b0, b0 + m) // per)
            par = parents[pidx]
            cand = par + rng.standard_normal((m, 2)) @ chol_w.T
            log_lr = -0.5 * (_quad(cand, mu, prec_b) - q_par[pidx])
            acc = np.log(rng.random(m)) < np.minimum(0.0, log_lr)
            cells = np.where(acc[:, None], cand, par)
            td = death_times(np.exp(cells[:, 0]), np.exp(cells[:, 1]), rng.random(m))
            pool, seen = _reservoir_update(pool, seen, cells[td > th], rng, st.max_parents)
        log_surv[cyc] = math.log10(seen / total) if seen > 0 else -np.inf
        if seen == 0:
            return dict(chain=chain_id, posterior_id=pid, start=start, clone=np.full(2, np.nan),
                        log_surv=log_surv, alive=False)
        parents = pool
    clone = parents[rng.integers(len(parents))]
    return dict(chain=chain_id, posterior_id=pid, start=start, clone=clone, log_surv=log_surv, alive=True)


_POST: dict = {}


def _init(post):
    global _POST
    _POST = post


def _one(i, st, seed):
    return run_whole_population_chain(i, _POST, st, seed)


def simulate_whole_population(post: Posterior | dict, st: WholePopSettings, n_chains: int = 100,
                              base_seed: int = 2026, n_workers: Optional[int] = None,
                              cache: Optional[str | Path] = None, progress: bool = True) -> pd.DataFrame:
    """Run n_chains lineages in parallel; one row per lineage."""
    if cache is not None and Path(cache).exists():
        df = pd.read_pickle(cache)
        if df.attrs.get("settings") == asdict(st) and df.attrs.get("seed") == base_seed and len(df) == n_chains:
            return df
    pdict = post.as_dict() if isinstance(post, Posterior) else post
    from .sim import default_workers, _maybe_tqdm
    n_workers = default_workers() if n_workers is None else int(n_workers)
    rows = []
    if n_workers == 1:
        _init(pdict)
        for i in _maybe_tqdm(range(n_chains), n_chains, progress):
            rows.append(_one(i, st, base_seed))
    else:
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init, initargs=(pdict,)) as ex:
            futs = [ex.submit(_one, i, st, base_seed) for i in range(n_chains)]
            for f in _maybe_tqdm(as_completed(futs), len(futs), progress):
                rows.append(f.result())
    rows.sort(key=lambda r: r["chain"])
    df = pd.DataFrame({
        "chain": [r["chain"] for r in rows],
        "posterior_id": [r["posterior_id"] for r in rows],
        "alive": [r["alive"] for r in rows],
        "start_lnd": [r["start"][0] for r in rows], "start_lnp": [r["start"][1] for r in rows],
        "clone_lnd": [r["clone"][0] for r in rows], "clone_lnp": [r["clone"][1] for r in rows],
    })
    for c in range(st.n_cycles):
        df[f"log_surv_cycle{c + 1}"] = [r["log_surv"][c] for r in rows]
    df.attrs["settings"] = asdict(st)
    df.attrs["seed"] = base_seed
    if cache is not None:
        df.to_pickle(cache)
    return df


def gain_table(df: pd.DataFrame, t_test: float) -> dict:
    """Reduction of the starting strain and of the final clone under t_test, and the difference."""
    a = df[df["alive"]]
    if len(a) == 0:                       # every lineage was lost
        nan = float("nan")
        return dict(R0_median=nan, Rk_median=nan, gain_median=nan, gain_q25=nan, gain_q75=nan,
                    share_alive=0.0, n_alive=0)
    R0 = log_reduction(t_test, np.exp(a["start_lnd"]), np.exp(a["start_lnp"]))
    R1 = log_reduction(t_test, np.exp(a["clone_lnd"]), np.exp(a["clone_lnp"]))
    g = np.asarray(R0 - R1)
    return dict(R0_median=float(np.median(R0)), Rk_median=float(np.median(R1)),
                gain_median=float(np.median(g)), gain_q25=float(np.quantile(g, .25)),
                gain_q75=float(np.quantile(g, .75)), share_alive=float(df["alive"].mean()),
                n_alive=int(len(a)))
