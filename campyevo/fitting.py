"""Fitting the multilevel model with CmdStan (via cmdstanpy)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAN_DIR = ROOT / "stan"


def compile_model(name: str = "multilevel_weibull", force: bool = False):
    from cmdstanpy import CmdStanModel
    stan_file = STAN_DIR / f"{name}.stan"
    return CmdStanModel(stan_file=str(stan_file), force_compile=force,
                        cpp_options={"STAN_THREADS": False})


def jittered_inits(base: dict, chains: int, seed: int = 1, scale: float = 0.1) -> list[dict]:
    """One set of starting values per chain: `base` with small random perturbations,
    so that chains do not start from the same point (needed for a meaningful R-hat)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(chains):
        d = {}
        for k, v in base.items():
            a = np.asarray(v, float)
            if k.startswith("Lcorr"):
                d[k] = a.tolist()
            elif k.startswith("sigma"):
                d[k] = (a * np.exp(rng.normal(0, scale, a.shape))).tolist()
            elif k in ("logC0", "theta_curve"):
                d[k] = (a + rng.normal(0, scale / 5, a.shape)).tolist()
            else:
                d[k] = (a + rng.normal(0, scale, a.shape)).tolist() if a.ndim else float(a + rng.normal(0, scale))
        out.append(d)
    return out


def fit(data: dict, model=None, chains: int = 4, iter_warmup: int = 2000,
        iter_sampling: int = 2500, adapt_delta: float = 0.95, max_treedepth: int = 12,
        seed: int = 12345, inits=None, output_dir: str | Path | None = None,
        show_progress: bool = True, parallel_chains: int | None = None, **kw):
    """Run NUTS. Defaults give 10,000 posterior draws (4 x 2,500).

    inits: a dict (jittered per chain automatically) or a list of dicts.
    """
    model = model or compile_model()
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(inits, dict):
        inits = jittered_inits(inits, chains, seed)
    return model.sample(data=data, chains=chains, parallel_chains=parallel_chains or chains,
                        iter_warmup=iter_warmup, iter_sampling=iter_sampling,
                        adapt_delta=adapt_delta, max_treedepth=max_treedepth, seed=seed,
                        inits=inits, output_dir=str(output_dir) if output_dir else None,
                        show_progress=show_progress, **kw)


KEY_VARS = ["logdelta_mean", "logpower_mean", "sigma_inter", "sigma_intra",
            "corr_inter", "corr_intra"]


def diagnostics(fit, key_vars=KEY_VARS) -> dict:
    """Divergences, tree-depth saturation, R-hat and ESS of the key quantities."""
    summ = fit.summary()
    rows = [r for r in summ.index if any(r == v or r.startswith(v + "[") for v in key_vars)]
    key = summ.loc[rows]
    sd = fit.method_variables()
    div = int(np.sum(sd["divergent__"]))
    td = int(np.sum(sd["treedepth__"] >= fit.metadata.cmdstan_config.get("max_depth", 10)))
    rhat_col = "R_hat" if "R_hat" in summ.columns else [c for c in summ.columns if "hat" in c][0]
    ess_col = [c for c in summ.columns if c.startswith("ESS_bulk") or c == "N_Eff"][0]
    all_rhat = summ[rhat_col].replace([np.inf, -np.inf], np.nan).dropna()
    return dict(
        key_summary=key,
        divergences=div,
        treedepth_hits=td,
        max_rhat_all=float(all_rhat.max()),
        min_ess_key=float(key[ess_col].min()),
        ok=(div == 0 and all_rhat.max() < 1.01),
    )


def strain_table(fit, strains: list[str]) -> pd.DataFrame:
    """Posterior means of each strain's (delta, p)."""
    th = fit.stan_variable("theta_strain")          # (draws, S, 2)
    d, p = np.exp(th[:, :, 0]), np.exp(th[:, :, 1])
    return pd.DataFrame({
        "strain": strains,
        "delta (median)": np.median(d, axis=0),
        "delta 2.5%": np.quantile(d, 0.025, axis=0),
        "delta 97.5%": np.quantile(d, 0.975, axis=0),
        "p (median)": np.median(p, axis=0),
        "p 2.5%": np.quantile(p, 0.025, axis=0),
        "p 97.5%": np.quantile(p, 0.975, axis=0),
    })
