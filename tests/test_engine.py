"""Checks of the simulation engine.

Run with  `python -m pytest tests -q`  (about 2-5 minutes on a laptop), or
`python tests/test_engine.py` for a printed report.

1. fast vs reference engine: same distribution of the parameters after k
   cycles and same extinction probability (two-sample KS tests and a
   two-proportion test).
2. no heating: the chain relaxes to MVN_inter (Mahalanobis^2 ~ chi2(2)).
3. the heating times 2/4/6 delta give 2/4/6-log reductions at p = 1.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campyevo.posterior import Posterior          # noqa: E402
from campyevo.sim import SimSettings, simulate, log_reduction, ATCC33560   # noqa: E402
from campyevo.analysis import stationary_check    # noqa: E402

MU = np.array([0.259, 0.0053])
COV_B = np.array([[0.06659, -0.03597], [-0.03597, 0.09372]])
COV_W = np.array([[0.10 ** 2, 0.0], [0.0, 0.12 ** 2]])     # placeholder intra-strain values


def toy_posterior(n=200, seed=0):
    """Posterior-like draws scattered a little around fixed values."""
    rng = np.random.default_rng(seed)
    mu = MU + rng.normal(0, 0.02, (n, 2))
    s = np.exp(rng.normal(0, 0.05, (n, 1, 1)))
    return Posterior(mu, COV_B[None] * s, COV_W[None] * s, source="toy")


def _compare(t_heating, n_cycles=15, n_chains=1500, n_cell=10 ** 4, workers=2):
    post = toy_posterior()
    out = {}
    for eng in ("fast", "reference"):
        st = SimSettings(t_heating=t_heating, n_cycles=n_cycles, n_cell=n_cell, engine=eng)
        out[eng] = simulate(post, st, n_chains=n_chains, base_seed=11 if eng == "fast" else 99,
                            n_workers=workers, progress=False)
    f, r = out["fast"], out["reference"]
    a, b = f.final_params(), r.final_params()
    ks_d = stats.ks_2samp(a[:, 0], b[:, 0]).pvalue if len(a) > 20 and len(b) > 20 else 1.0
    ks_p = stats.ks_2samp(a[:, 1], b[:, 1]).pvalue if len(a) > 20 and len(b) > 20 else 1.0
    sf, sr = f.survival_ratio()[-1], r.survival_ratio()[-1]
    pooled = (sf + sr) / 2
    se = np.sqrt(max(pooled * (1 - pooled), 1e-12) * 2 / n_chains)
    p_surv = 2 * stats.norm.sf(abs(sf - sr) / se) if se > 0 else 1.0
    return dict(ks_lndelta=ks_d, ks_lnp=ks_p, surv_fast=sf, surv_ref=sr, p_surv=p_surv,
                mean_fast=a.mean(axis=0) if len(a) else None, mean_ref=b.mean(axis=0) if len(b) else None)


def test_heating_times():
    for R in (2, 4, 6):
        assert np.isclose(log_reduction(R * 1.3, 1.3, 1.0), R)
    assert np.isclose(log_reduction(6 * 1.3, 1.3, 1.7), 6.0)      # 6 delta gives 6 log for any p


def test_fast_equals_reference_2log():
    r = _compare(2 * np.exp(MU[0]))
    assert min(r["ks_lndelta"], r["ks_lnp"]) > 0.001, r
    assert r["p_surv"] > 0.001, r


def test_fast_equals_reference_6log():
    r = _compare(6 * np.exp(MU[0]))
    assert min(r["ks_lndelta"], r["ks_lnp"]) > 0.001, r
    assert r["p_surv"] > 0.001, r


def test_no_heating_recovers_target():
    post = toy_posterior()
    pid = post.central_draw()
    st = SimSettings(t_heating=0.0, n_cycles=300, init="fixed", init_logpara=ATCC33560,
                     posterior_id=pid)
    res = simulate(post, st, n_chains=1500, n_workers=2, progress=False)
    chk = stationary_check(res, post.mu_inter[pid], post.cov_inter[pid])
    assert chk["ks_p"] > 0.001, chk["table"]


if __name__ == "__main__":
    import time
    test_heating_times()
    for name, th in [("2-log", 2 * np.exp(MU[0])), ("4-log", 4 * np.exp(MU[0])), ("6-log", 6 * np.exp(MU[0]))]:
        t0 = time.time()
        r = _compare(th)
        print(f"{name}: KS p (ln delta) = {r['ks_lndelta']:.3f}, KS p (ln p) = {r['ks_lnp']:.3f}, "
              f"survival fast {r['surv_fast']:.3f} vs reference {r['surv_ref']:.3f} (p = {r['p_surv']:.3f}); "
              f"means fast {np.round(r['mean_fast'], 3)} vs ref {np.round(r['mean_ref'], 3)}  [{time.time() - t0:.0f} s]")
    t0 = time.time()
    test_no_heating_recovers_target()
    print(f"no-heating stationary check passed [{time.time() - t0:.0f} s]")
