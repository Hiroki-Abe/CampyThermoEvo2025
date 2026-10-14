"""Heating-regrowth simulation (Metropolis-Hastings random walk + stochastic thermal death).

One cycle, for a population carried over from the previous cycle:

  1. Regrowth. The carried-over cell(s) regrow to N_cell cells. Each daughter
     proposes a parameter couple (ln delta, ln p) from MVN_intra centred on its
     parent (intra-strain divergence).
  2. Constraint. The proposal is accepted with probability
     min(1, LR^L), LR = f_MVN_inter(candidate) / f_MVN_inter(parent);
     otherwise the daughter keeps the parent's couple. At L = 1 this is a
     standard random-walk Metropolis step whose stationary distribution is
     MVN_inter.
  3. Heating. Each cell draws its own death time from the modified Weibull
     model; it survives if the death time exceeds the heating time.
  4. Carry-over. n surviving cells (sampled with replacement) start the next
     cycle. No survivor = extinction of the lineage (NaN from then on).

Two engines are provided.

``reference``  draws all N_cell cells every cycle, exactly as in the published
               code (``heat_rev`` in the original notebook, ``heat_rev_general``
               in revision_core.py). Works for any n.

``fast``       for n = 1 only. It draws cells in blocks and stops at the first
               survivor. This is the SAME probability model, not an
               approximation: the cells are i.i.d., so given which of them
               survive, the survivors are i.i.d. draws from the distribution
               conditional on survival, and the first survivor has the same
               distribution as a survivor chosen at random. Extinction happens
               exactly when none of the N_cell cells survives. The cost per
               cycle is about min(N_cell, 1 / P(survival)) cells instead of
               N_cell, which is what makes N_cell = 10^6 practical.
               ``tests/test_engine.py`` checks the two engines against each other.

Random numbers: chain k uses ``np.random.default_rng([base_seed, k])`` so a
result does not depend on the number of worker processes. The two engines use
the random stream differently, so they agree in distribution, not path by path.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence, Union

import multiprocessing as mp
import numpy as np
import pandas as pd

from .posterior import Posterior

LN10 = math.log(10.0)
ATCC33560 = (math.log(1.80), math.log(1.17))   # (ln delta0, ln p0) of the type strain

# ---------------------------------------------------------------------------
# Modified Weibull helpers
# ---------------------------------------------------------------------------


def log_reduction(t, delta, p):
    """log10 reduction after heating for t: 6 * (t / (6 delta))^p."""
    t = np.asarray(t, float)
    return 6.0 * (t / (6.0 * np.asarray(delta, float))) ** np.asarray(p, float)


def time_for_reduction(R, delta, p):
    """Heating time that gives a log10 reduction of R: 6 delta (R/6)^(1/p)."""
    return 6.0 * np.asarray(delta, float) * (np.asarray(R, float) / 6.0) ** (1.0 / np.asarray(p, float))


def death_times(delta, p, u):
    """Inverse-transform sample of single-cell death times, u ~ U(0, 1)."""
    return 6.0 * delta * ((-np.log1p(-u) / LN10) / 6.0) ** (1.0 / p)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NCellSpec = Union[int, tuple]


@dataclass(frozen=True)
class SimSettings:
    """Everything that defines one simulation condition.

    n_cell
        Population size after regrowth. An int (published: 10**4), or
        ("log10normal", mean, sd[, max]) to draw log10 N_cell ~ Normal(mean, sd)
        afresh in every cycle (N_cell = max(1, round(10**x)), capped at max,
        default 10**7). The original notebook contains this variant, commented
        out, with mean 4 and sd 1.
    s_intra, s_inter
        Multipliers on the intra- and inter-strain STANDARD DEVIATIONS (the
        covariance matrices are multiplied by the square). Published: 1, 1.
    n_carry
        Cells carried over to the next cycle. Published: 1.
    L
        Exponent on the density ratio in the acceptance step. Published: 1.
    init
        "inter": start from a draw of MVN_inter; "fixed": start from init_logpara.
    posterior_id
        None (published): each chain uses a random posterior draw, so parameter
        uncertainty is propagated. An int fixes the draw (algorithm validation).
    engine
        "auto" (fast when n_carry == 1), "fast" or "reference".
    """

    t_heating: float
    n_cycles: int = 100
    n_cell: NCellSpec = 10 ** 4
    s_intra: float = 1.0
    s_inter: float = 1.0
    n_carry: int = 1
    L: float = 1.0
    init: str = "inter"
    init_logpara: Optional[tuple] = None
    posterior_id: Optional[int] = None
    engine: str = "auto"

    def resolved_engine(self) -> str:
        if self.engine == "auto":
            return "fast" if self.n_carry == 1 else "reference"
        if self.engine == "fast" and self.n_carry != 1:
            raise ValueError("the fast engine is exact only for n_carry = 1")
        return self.engine

    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


def _draw_ncell(spec: NCellSpec, rng: np.random.Generator) -> int:
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    kind = spec[0]
    if kind == "log10normal":
        mean, sd = float(spec[1]), float(spec[2])
        cap = int(spec[3]) if len(spec) > 3 else 10 ** 7
        return int(min(max(1, round(10 ** rng.normal(mean, sd))), cap))
    raise ValueError(f"unknown n_cell specification: {spec!r}")


# ---------------------------------------------------------------------------
# One chain
# ---------------------------------------------------------------------------


def _chain_setup(post: dict, st: SimSettings, rng):
    n_post = post["mu_inter"].shape[0]
    pid = int(rng.integers(n_post)) if st.posterior_id is None else int(st.posterior_id)
    mu = post["mu_inter"][pid]
    cov_b = post["cov_inter"][pid] * st.s_inter ** 2
    cov_w = post["cov_intra"][pid] * st.s_intra ** 2
    chol_w = np.linalg.cholesky(cov_w)
    prec_b = np.linalg.inv(cov_b)
    if st.init == "inter":
        state = rng.multivariate_normal(mu, cov_b, size=st.n_carry)
    elif st.init == "fixed":
        if st.init_logpara is None:
            raise ValueError("init='fixed' requires init_logpara")
        state = np.tile(np.asarray(st.init_logpara, float), (st.n_carry, 1))
    else:
        raise ValueError("init must be 'inter' or 'fixed'")
    return pid, mu, chol_w, prec_b, state


def _quad(x, mu, prec):
    """(x - mu)' P (x - mu) row-wise; -0.5 * this is ln f_MVN up to a constant."""
    d = x - mu
    return np.einsum("...i,ij,...j->...", d, prec, d)


def run_chain(chain_id: int, post: dict, st: SimSettings, base_seed: int = 2026):
    """Run one replicate chain. Returns (trajectory (n_cycles+1, 2), posterior id).

    The trajectory holds the mean (ln delta, ln p) of the carried-over cells
    (the single carried cell when n_carry = 1); NaN after extinction.
    """
    rng = np.random.default_rng([int(base_seed), int(chain_id)])
    pid, mu, chol_w, prec_b, state = _chain_setup(post, st, rng)
    traj = np.full((st.n_cycles + 1, 2), np.nan)
    traj[0] = state.mean(axis=0)
    engine = st.resolved_engine()
    th = float(st.t_heating)
    L = float(st.L)

    for cyc in range(1, st.n_cycles + 1):
        n_cell = _draw_ncell(st.n_cell, rng)

        if engine == "fast":
            parent = state[0]
            q_par = _quad(parent, mu, prec_b)
            remaining, block, found = n_cell, 64, None
            while remaining > 0:
                m = min(block, remaining)
                cand = parent + rng.standard_normal((m, 2)) @ chol_w.T
                log_lr = -0.5 * L * (_quad(cand, mu, prec_b) - q_par)
                acc = np.log(rng.random(m)) < np.minimum(0.0, log_lr)
                cells = np.where(acc[:, None], cand, parent)
                td = death_times(np.exp(cells[:, 0]), np.exp(cells[:, 1]), rng.random(m))
                hit = np.flatnonzero(td > th)
                if hit.size:
                    found = cells[hit[0]]
                    break
                remaining -= m
                block = min(block * 8, 1 << 16)
            if found is None:
                break                                   # extinction
            state = found[None, :]

        else:  # reference: whole population, as in the published code
            per = max(n_cell // st.n_carry, 1)
            parents = np.repeat(state, per, axis=0)
            m = parents.shape[0]
            cand = parents + rng.standard_normal((m, 2)) @ chol_w.T
            log_lr = -0.5 * L * (_quad(cand, mu, prec_b) - _quad(parents, mu, prec_b))
            acc = np.log(rng.random(m)) < np.minimum(0.0, log_lr)
            cells = np.where(acc[:, None], cand, parents)
            td = death_times(np.exp(cells[:, 0]), np.exp(cells[:, 1]), rng.random(m))
            surv = np.flatnonzero(td > th)
            if surv.size == 0:
                break
            state = cells[rng.choice(surv, size=st.n_carry, replace=True)]

        traj[cyc] = state.mean(axis=0)
    return traj, pid


# ---------------------------------------------------------------------------
# Many chains in parallel
# ---------------------------------------------------------------------------

_POST: dict = {}


def _init_worker(post: dict):
    global _POST
    _POST = post


def _run_chunk(ids: Sequence[int], st: SimSettings, base_seed: int):
    out = np.empty((len(ids), st.n_cycles + 1, 2))
    pids = np.empty(len(ids), dtype=np.int64)
    for i, k in enumerate(ids):
        out[i], pids[i] = run_chain(k, _POST, st, base_seed)
    return np.asarray(ids), out, pids


@dataclass
class SimResult:
    traj: np.ndarray            # (n_chains, n_cycles+1, 2): ln delta, ln p
    posterior_ids: np.ndarray   # (n_chains,)
    settings: SimSettings
    base_seed: int
    meta: dict = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return self.traj.shape[0]

    @property
    def alive(self) -> np.ndarray:
        """(n_chains, n_cycles+1) True while the lineage survives."""
        return ~np.isnan(self.traj[:, :, 0])

    def survival_ratio(self) -> np.ndarray:
        return self.alive.mean(axis=0)

    def summary(self) -> pd.DataFrame:
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d, p = self.traj[:, :, 0], self.traj[:, :, 1]
                return pd.DataFrame({
                    "cycle": np.arange(self.traj.shape[1]),
                    "lndelta_mean": np.nanmean(d, axis=0),
                    "lnp_mean": np.nanmean(p, axis=0),
                    "lndelta_sd": np.nanstd(d, axis=0, ddof=1),
                    "lnp_sd": np.nanstd(p, axis=0, ddof=1),
                    "survival_ratio": self.survival_ratio(),
                })

    def final_params(self, cycle: int = -1) -> np.ndarray:
        x = self.traj[:, cycle, :]
        return x[~np.isnan(x[:, 0])]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        st = asdict(self.settings)
        np.savez_compressed(path, traj=self.traj, posterior_ids=self.posterior_ids,
                            base_seed=self.base_seed,
                            settings=np.array(repr(st)), meta=np.array(repr(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "SimResult":
        import ast
        z = np.load(path, allow_pickle=False)
        st = ast.literal_eval(str(z["settings"]))
        if isinstance(st.get("n_cell"), list):
            st["n_cell"] = tuple(st["n_cell"])
        if isinstance(st.get("init_logpara"), list):
            st["init_logpara"] = tuple(st["init_logpara"])
        return cls(z["traj"], z["posterior_ids"], SimSettings(**st), int(z["base_seed"]),
                   ast.literal_eval(str(z["meta"])))


def default_workers() -> int:
    n = os.cpu_count() or 2
    return max(1, n - 1)


def simulate(post: Posterior | dict, settings: SimSettings, n_chains: int = 10_000,
             base_seed: int = 2026, n_workers: Optional[int] = None,
             chunk: Optional[int] = None, progress: bool = True,
             cache: Optional[str | Path] = None) -> SimResult:
    """Run ``n_chains`` independent chains of one condition in parallel.

    cache: if given and the file exists with the same settings, it is loaded
    instead of re-running; otherwise the result is saved there.
    """
    if cache is not None and Path(cache).exists():
        res = SimResult.load(cache)
        if res.settings == settings and res.n_chains == n_chains and res.base_seed == base_seed:
            return res
    pdict = post.as_dict() if isinstance(post, Posterior) else post
    n_workers = default_workers() if n_workers is None else int(n_workers)
    ids = np.arange(n_chains)
    if chunk is None:
        chunk = max(1, min(250, math.ceil(n_chains / (n_workers * 8))))
    chunks = [ids[i:i + chunk] for i in range(0, n_chains, chunk)]

    traj = np.empty((n_chains, settings.n_cycles + 1, 2))
    pids = np.empty(n_chains, dtype=np.int64)

    if n_workers == 1:
        _init_worker(pdict)
        it = (_run_chunk(c, settings, base_seed) for c in chunks)
        it = _maybe_tqdm(it, len(chunks), progress)
        for ii, tr, pp in it:
            traj[ii], pids[ii] = tr, pp
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(pdict,)) as ex:
            futs = [ex.submit(_run_chunk, c, settings, base_seed) for c in chunks]
            from concurrent.futures import as_completed
            for f in _maybe_tqdm(as_completed(futs), len(futs), progress):
                ii, tr, pp = f.result()
                traj[ii], pids[ii] = tr, pp

    res = SimResult(traj, pids, settings, base_seed,
                    meta=dict(n_workers=n_workers, engine=settings.resolved_engine()))
    if cache is not None:
        res.save(cache)
    return res


def _maybe_tqdm(it, total, progress):
    if not progress:
        return it
    try:
        from tqdm.auto import tqdm
        return tqdm(it, total=total, leave=False)
    except Exception:
        return it
