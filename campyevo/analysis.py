"""Summaries computed from simulation results (tables for the manuscript)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .sim import SimResult, log_reduction


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def plateau(summary: pd.DataFrame, col: str = "lndelta_mean", last: int = 20) -> float:
    """Mean of `col` over the last `last` cycles (the converged value)."""
    return float(summary[col].iloc[-last:].mean())


def cycles_to_fraction(summary: pd.DataFrame, col: str = "lndelta_mean",
                       frac: float = 0.95, last: int = 20, min_shift: float = 0.02):
    """First cycle from which `col` stays beyond `frac` of its total shift.

    The total shift is plateau - value at cycle 0. More robust than "within a
    fixed tolerance of the final value", which is dominated by Monte Carlo
    noise when the plateau is flat. Returns NaN when the total shift is below
    `min_shift` (nothing to converge).
    """
    v = summary[col].to_numpy()
    v0, vp = v[0], plateau(summary, col, last)
    shift = vp - v0
    if not np.isfinite(shift) or abs(shift) < min_shift:
        return np.nan
    progress = (v - v0) / shift
    ok = progress >= frac
    for i in range(len(v)):
        if ok[i:].all():
            return int(summary["cycle"].iloc[i])
    return np.nan


def condition_table(results: dict, labels: dict | None = None, cycle_shift_ref: int = 0) -> pd.DataFrame:
    """One row per condition: start, converged value, shift, cycles to 95 %, survival."""
    rows = []
    for key, res in results.items():
        sm = res.summary()
        row = dict(condition=labels.get(key, key) if labels else key)
        row.update({
            "ln delta (0)": sm["lndelta_mean"].iloc[cycle_shift_ref],
            "ln delta (plateau)": plateau(sm, "lndelta_mean"),
            "ln p (0)": sm["lnp_mean"].iloc[cycle_shift_ref],
            "ln p (plateau)": plateau(sm, "lnp_mean"),
            "shift ln delta": plateau(sm, "lndelta_mean") - sm["lndelta_mean"].iloc[cycle_shift_ref],
            "cycles to 50% (ln delta)": cycles_to_fraction(sm, "lndelta_mean", frac=0.5),
            "cycles to 95% (ln delta)": cycles_to_fraction(sm, "lndelta_mean"),
            "SD ln delta (final)": sm["lndelta_sd"].iloc[-1],
            "survival ratio (final)": sm["survival_ratio"].iloc[-1],
            "chains": res.n_chains,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Algorithm validation (no heating -> stationary distribution = MVN_inter)
# ---------------------------------------------------------------------------

def stationary_check(res: SimResult, mu: np.ndarray, cov: np.ndarray, cycle: int = -1) -> dict:
    X = res.final_params(cycle)
    emp_mu, emp_cov = X.mean(axis=0), np.cov(X.T)
    Xc = X - mu
    md2 = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(cov), Xc)
    ks = stats.kstest(md2, "chi2", args=(2,))
    corr = lambda c: c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    table = pd.DataFrame({
        "recovered": [emp_mu[0], emp_mu[1], np.sqrt(emp_cov[0, 0]), np.sqrt(emp_cov[1, 1]), corr(emp_cov)],
        "target": [mu[0], mu[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), corr(cov)],
    }, index=["mean ln delta", "mean ln p", "SD ln delta", "SD ln p", "corr"])
    return dict(table=table, md2=md2, ks_D=float(ks.statistic), ks_p=float(ks.pvalue), n=len(X))


# ---------------------------------------------------------------------------
# Survival curves implied by a set of parameter couples (Fig. 6)
# ---------------------------------------------------------------------------

def survival_band(params: np.ndarray, times: np.ndarray, q=(0.005, 0.5, 0.995)) -> np.ndarray:
    """Quantiles of log10 N/N0 = -6 (t / 6 delta)^p over parameter couples.

    Returns array (len(q), len(times)).
    """
    d = np.exp(params[:, 0])[:, None]
    p = np.exp(params[:, 1])[:, None]
    logS = -log_reduction(times[None, :], d, p)
    return np.quantile(logS, q, axis=0)


# ---------------------------------------------------------------------------
# External benchmark: gain in log survival after k cycles at a fixed treatment
# ---------------------------------------------------------------------------

def gain_in_log_survival(res: SimResult, t_treat: float, cycle: int = 10) -> dict:
    """Median log10 reduction at t_treat for parameters at cycle 0 and at `cycle`.

    The gain (reduction at cycle 0 minus reduction at `cycle`, among lineages
    still alive) is the simulated counterpart of the increase in survival after
    a fixed heat treatment reported by Berdejo et al. (2024) after 10 cycles.
    """
    alive = res.alive[:, cycle]
    x0 = res.traj[alive, 0, :]
    xk = res.traj[alive, cycle, :]
    R0 = log_reduction(t_treat, np.exp(x0[:, 0]), np.exp(x0[:, 1]))
    Rk = log_reduction(t_treat, np.exp(xk[:, 0]), np.exp(xk[:, 1]))
    return dict(median_R0=float(np.median(R0)), median_Rk=float(np.median(Rk)),
                median_gain=float(np.median(R0 - Rk)),
                gain_q=np.quantile(R0 - Rk, [0.025, 0.25, 0.75, 0.975]),
                fold_delta=float(np.exp(np.median(xk[:, 0] - x0[:, 0]))),
                n=int(alive.sum()))


def gain_same_treatment(res: SimResult, t_test: float, cycle: int = 10) -> dict:
    """Berdejo-style comparison: each lineage's ancestor (cycle 0) and its descendant at
    `cycle` are given the same heat treatment t_test; the gain is the difference in log10
    reduction (ancestor minus descendant), among lineages still alive at `cycle`.

    With t_test equal to the treatment used in every cycle, this is the quantity Berdejo
    et al. (2024) measured: survival of the parental strain and of the evolved lineages
    under the selection treatment itself.
    """
    alive = res.alive[:, cycle]
    x0, xk = res.traj[alive, 0, :], res.traj[alive, cycle, :]
    R0 = log_reduction(t_test, np.exp(x0[:, 0]), np.exp(x0[:, 1]))
    Rk = log_reduction(t_test, np.exp(xk[:, 0]), np.exp(xk[:, 1]))
    g = R0 - Rk
    return dict(R0_median=float(np.median(R0)), Rk_median=float(np.median(Rk)),
                gain_median=float(np.median(g)), gain_q25=float(np.quantile(g, .25)),
                gain_q75=float(np.quantile(g, .75)), gain_mean=float(g.mean()),
                share_alive=float(alive.mean()), n_alive=int(alive.sum()))


# ---------------------------------------------------------------------------
# QMRA illustration
# ---------------------------------------------------------------------------

def beta_poisson(dose, alpha: float = 0.145, beta: float = 7.59):
    """Approximate Beta-Poisson dose-response. alpha=0.145, beta=7.59 are the
    values of Medema et al. (1996) for infection by C. jejuni; check that the
    reference cited in the manuscript matches the values used."""
    return 1.0 - (1.0 + np.asarray(dose, float) / beta) ** (-alpha)


def risk_for_params(params: np.ndarray, t_process: float, log10_N0: float = 2.0,
                    alpha: float = 0.145, beta: float = 7.59):
    R = log_reduction(t_process, np.exp(params[:, 0]), np.exp(params[:, 1]))
    dose = 10.0 ** (log10_N0 - R)
    return beta_poisson(dose, alpha, beta), dose


def baseline_strain_draws(post, n: int = 20_000, seed: int = 0) -> np.ndarray:
    """Present-day strains: one MVN_inter draw per randomly chosen posterior draw."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, post.n_draws, n)
    return np.array([rng.multivariate_normal(post.mu_inter[i], post.cov_inter[i]) for i in idx])


# ---------------------------------------------------------------------------
# Readouts in microbiological units (for the main figures)
# ---------------------------------------------------------------------------

def reduction_over_cycles(res: SimResult, t_treat: float, q=(0.25, 0.5, 0.75)) -> pd.DataFrame:
    """log10 reduction that the treatment t_treat achieves on each surviving lineage, per cycle.

    Returns one row per cycle with the quantiles over lineages alive at that cycle.
    """
    d = np.exp(res.traj[:, :, 0])
    p = np.exp(res.traj[:, :, 1])
    with np.errstate(invalid="ignore"):
        R = log_reduction(t_treat, d, p)                 # (chains, cycles); NaN after extinction
    out = {"cycle": np.arange(R.shape[1]), "n_alive": np.sum(~np.isnan(R), axis=0)}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for qq in q:
            out[f"q{int(qq * 100)}"] = np.nanquantile(R, qq, axis=0)
    return pd.DataFrame(out)


def curve_means(post):
    """Posterior means of each curve's (ln delta, ln p) and log10 N0, if stored.

    Returns (theta (J, 2), logC0 (J,)) or (None, None). Curve j corresponds to
    row j + 1 of CurveData.curves.
    """
    ex = getattr(post, "extra", {}) or {}
    th, c0 = ex.get("theta_curve"), ex.get("logC0")
    if th is None:
        return None, None
    th = np.asarray(th).mean(axis=0)
    c0 = None if c0 is None else np.asarray(c0).mean(axis=0)
    return th, c0


def strain_means(post) -> np.ndarray | None:
    """Posterior means of the 50 measured strains' (ln delta, ln p), if stored."""
    th = post.extra.get("theta_strain") if hasattr(post, "extra") else None
    if th is None:
        return None
    th = np.asarray(th)
    return th.mean(axis=0) if th.shape[-1] == 2 else th.mean(axis=0).T
