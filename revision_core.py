"""
revision_core.py
================
Generalised simulation engine for the MRA revision (MRAN-D-26-00103).

This module re-implements `heat_rev()` from the original notebook with every
quantity that a reviewer asked about exposed as an explicit argument:

    sd_scale_within   -> Reviewer #3 (main request) / Reviewer #2 (L156)
    sd_scale_between  -> Reviewer #2 (L156)
    n_carry           -> Reviewer #2 (L214)  [carry-over parameter n]
    L_para            -> Reviewer #3 (likelihood-ratio exponent found in the code)
    init_mode         -> Reviewer #3 (purpose of the two initialisation modes)
    posterior_id      -> fix the Bayesian draw (needed for algorithm validation)

Setting
    sd_scale_within = 1, sd_scale_between = 1, n_carry = 1,
    L_para = 1, init_mode = 'inter', posterior_id = None
reproduces the published simulation EXACTLY (verified against the original code).

--------------------------------------------------------------------------
MODEL (unchanged from the paper; notation clarified)
--------------------------------------------------------------------------
    log10 N_t = log10 N_0  -  6 * ( t / (6*delta) )**p

    * The time to reach a 6-log reduction is  t = 6*delta  (for ANY p).
    * Hence  delta  is the MEAN CHARACTERISTIC TIME PER LOG-REDUCTION,
      averaged over a 6-log reduction:   delta = delta_6 / 6.
    * With p = 1 the heating times 2*delta, 4*delta, 6*delta give EXACTLY
      2-, 4- and 6-log reductions.  This is why those heating times were used.

    Inverse-transform sampling of individual-cell death time (u ~ U(0,1)):
        t_death = 6*delta * ( -log10(1-u) / 6 )**(1/p)
    -> identical to the original code.  Verified.
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as _mvn


# =========================================================================
# 1.  Deterministic helpers (used for Part 1 of the revision)
# =========================================================================

def log_reduction(t, delta, p):
    """Log10 reduction achieved at time t by a cell with parameters (delta, p)."""
    t = np.asarray(t, dtype=float)
    return 6.0 * (t / (6.0 * np.asarray(delta, dtype=float))) ** np.asarray(p, dtype=float)


def time_for_reduction(target_log, delta, p):
    """Time needed to achieve `target_log` log10 reduction.  Generalises the
    6-log convention to ANY target (Reviewer #3 asked for this generalisation)."""
    return 6.0 * delta * (np.asarray(target_log, dtype=float) / 6.0) ** (1.0 / np.asarray(p, dtype=float))


def death_time_sample(delta, p, rng):
    """Inverse-transform sample of individual-cell death time."""
    u = rng.uniform(0.0, 1.0, size=np.shape(delta))
    return 6.0 * delta * (-np.log10(1.0 - u) / 6.0) ** (1.0 / p)


# =========================================================================
# 2.  Generalised chain
# =========================================================================

def heat_rev_general(seed_ID,
                     samples,
                     t_heating,
                     revo_num=100,
                     cell_count=10 ** 4,
                     sd_scale_within=1.0,
                     sd_scale_between=1.0,
                     n_carry=1,
                     L_para=1.0,
                     init_mode='inter',
                     init_logpara=None,
                     posterior_id=None,
                     return_full=False):
    """
    Run ONE replicate chain of `revo_num` heating-regrowth cycles.

    Parameters
    ----------
    seed_ID : int
        RNG seed (also used to label the output columns).
    samples : dict
        Stan posterior samples: needs 'Paras0_between', 'cov_between', 'cov_within'.
    t_heating : float
        Heating duration per cycle (min).  0 => no heating (pure MH chain).
    revo_num : int
        Number of heating-regrowth cycles.
    cell_count : int
        Population size at the end of regrowth, i.e. N_cell (default 1e4 = 4-log).
    sd_scale_within : float
        Multiplier applied to the intra-strain STANDARD DEVIATIONS.
        (COV_intra is scaled by sd_scale_within**2.)  1.0 = published setting.
    sd_scale_between : float
        Multiplier applied to the inter-strain STANDARD DEVIATIONS.
        (COV_inter is scaled by sd_scale_between**2.) 1.0 = published setting.
    n_carry : int
        Carry-over parameter n: number of cells transferred to the next cycle.
        n_carry = 1 = published setting (maximal bottleneck).
    L_para : float
        Exponent applied to the likelihood ratio in the Metropolis filter,
        i.e. acceptance prob = min(1, LR**L_para).
        L_para = 1 => standard Metropolis-Hastings targeting MVN_inter.
        L_para = 0 => no inter-strain constraint (pure diffusion).
        L_para > 1 => stronger "species gravity".
    init_mode : {'inter', 'fixed'}
        'inter' : start from a random draw of MVN_inter (species-scale question)
        'fixed' : start from `init_logpara` (e.g. ATCC 33560; strain-scale question)
    init_logpara : array-like, shape (2,)
        [ln delta0, ln p0].  Required if init_mode == 'fixed'.
        ATCC 33560:  np.log([1.80, 1.17])
    posterior_id : int or None
        If None (default, = published behaviour) a posterior draw is chosen at
        random for this replicate, so Bayesian PARAMETER UNCERTAINTY is
        propagated across replicates (a 2-D Monte Carlo structure).
        If an int, that posterior draw is FIXED -- required for the algorithm
        validation in Part 2, where a single well-defined target distribution
        is needed.
    return_full : bool
        If True also return the full (n_carry, 2) state at every cycle.

    Returns
    -------
    pandas.DataFrame with columns ['logdelta <seed>', 'logpower <seed>'] and
    revo_num+1 rows.  The value stored is the MEAN over the n_carry carried-over
    lineages (identical to the single lineage when n_carry == 1).
    NaN marks complete inactivation (extinction) of the population.
    """
    rng = np.random.default_rng(seed_ID)

    n_post = samples['Paras0_between'].shape[0]
    pid = rng.integers(n_post) if posterior_id is None else int(posterior_id)

    logpara_mean = np.asarray(samples['Paras0_between'][pid], dtype=float)
    cov_between = np.asarray(samples['cov_between'][pid], dtype=float) * (sd_scale_between ** 2)
    cov_within = np.asarray(samples['cov_within'][pid], dtype=float) * (sd_scale_within ** 2)

    chol_within = np.linalg.cholesky(cov_within)
    target = _mvn(mean=logpara_mean, cov=cov_between, allow_singular=True)

    # ---- initial state: (n_carry, 2) --------------------------------------
    if init_mode == 'inter':
        state = rng.multivariate_normal(logpara_mean, cov_between, size=n_carry)
    elif init_mode == 'fixed':
        if init_logpara is None:
            raise ValueError("init_mode='fixed' requires init_logpara")
        state = np.tile(np.asarray(init_logpara, dtype=float), (n_carry, 1))
    else:
        raise ValueError("init_mode must be 'inter' or 'fixed'")

    hist = np.full((revo_num + 1, 2), np.nan)
    hist[0] = state.mean(axis=0)
    full = [state.copy()] if return_full else None

    # daughters produced per carried-over lineage during regrowth
    per_lineage = max(int(cell_count // n_carry), 1)
    n_cells = per_lineage * n_carry

    for cyc in range(1, revo_num + 1):

        # ---- (i) REGROWTH: each carried cell expands to `per_lineage` daughters
        parents = np.repeat(state, per_lineage, axis=0)          # (n_cells, 2)

        # ---- (ii) INTRA-STRAIN DIVERGENCE  (Metropolis-Hastings step) ------
        # proposal ~ MVN(parent, COV_intra); done in log-space for stability
        z = rng.standard_normal((n_cells, 2))
        cand = parents + z @ chol_within.T

        log_l_cand = target.logpdf(cand)
        log_l_par = target.logpdf(parents)
        log_LR = L_para * (log_l_cand - log_l_par)               # = log(LR**L_para)
        accept = np.log(rng.uniform(size=n_cells)) < np.minimum(0.0, log_LR)

        cells = np.where(accept[:, None], cand, parents)         # (n_cells, 2)

        # ---- (iii) STOCHASTIC THERMAL DEATH -------------------------------
        delta = np.exp(cells[:, 0])
        power = np.exp(cells[:, 1])
        t_death = death_time_sample(delta, power, rng)
        surv = np.flatnonzero(t_death > t_heating)

        # ---- (iv) CARRY-OVER: transfer n_carry cells to the next cycle -----
        if surv.size == 0:                                       # extinction
            hist[cyc:] = np.nan
            if return_full:
                full.append(np.full((n_carry, 2), np.nan))
            break

        # survivors regrow and n_carry cells are transferred (droplet / surface
        # contact) -> sampling WITH replacement from the survivor pool.
        pick = rng.choice(surv, size=n_carry, replace=True)
        state = cells[pick]

        hist[cyc] = state.mean(axis=0)
        if return_full:
            full.append(state.copy())

    df = pd.DataFrame(hist,
                      columns=[f'logdelta {seed_ID}', f'logpower {seed_ID}'],
                      index=range(revo_num + 1))
    return (df, full) if return_full else df


# =========================================================================
# 3.  Batch runner
# =========================================================================

def run_batch(samples, t_heating, sim_num=10000, processor_num=6, **kw):
    """Run `sim_num` independent chains in parallel and concatenate."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    f = partial(heat_rev_general, samples=samples, t_heating=t_heating, **kw)
    with ProcessPoolExecutor(processor_num) as ex:
        out = list(ex.map(f, range(sim_num)))
    return pd.concat(out, axis=1)


# =========================================================================
# 4.  Summaries
# =========================================================================

def summarise(hist, revo_num=100):
    """Extract per-cycle mean of ln(delta) / ln(p) and the survival ratio."""
    d = hist.filter(like='logdelta')
    p = hist.filter(like='logpower')
    sim_num = d.shape[1]
    return pd.DataFrame({
        'cycle': np.arange(revo_num + 1),
        'logdelta_mean': d.mean(axis=1).values,
        'logpower_mean': p.mean(axis=1).values,
        'logdelta_sd': d.std(axis=1).values,
        'logpower_sd': p.std(axis=1).values,
        'survival_ratio': (1.0 - d.isna().mean(axis=1)).values,
    })


def final_params(hist):
    """(N_surviving, 2) array of the final [ln delta, ln p] of surviving chains."""
    d = hist.filter(like='logdelta').iloc[-1].values
    p = hist.filter(like='logpower').iloc[-1].values
    ok = ~np.isnan(d)
    return np.column_stack([d[ok], p[ok]])


def convergence_cycle(summary, col='logdelta_mean', tol=0.01):
    """First cycle after which `col` stays within `tol` of its final value.
    Gives the QUANTITATIVE convergence rate the reviewers asked for."""
    v = summary[col].values
    final = v[-1]
    inside = np.abs(v - final) <= tol
    for i in range(len(v)):
        if inside[i:].all():
            return int(summary['cycle'].values[i])
    return int(summary['cycle'].values[-1])
