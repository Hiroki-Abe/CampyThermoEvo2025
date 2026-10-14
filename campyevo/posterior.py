"""Posterior draws of the species-level quantities used by the simulation.

The simulation needs, for every posterior draw i,

    mu_inter[i]   (ln delta_mean, ln p_mean)            = Paras0_between
    cov_inter[i]  COV_inter                             = cov_between
    cov_intra[i]  COV_intra                             = cov_within

They can come from a new CmdStan fit (:meth:`Posterior.from_cmdstan`) or from
the pickle of the original PyStan 2.19 fit (:meth:`Posterior.from_pystan_pickle`),
so the published results can be reproduced and compared with the refit.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Posterior:
    mu_inter: np.ndarray          # (n_draws, 2)
    cov_inter: np.ndarray         # (n_draws, 2, 2)
    cov_intra: np.ndarray         # (n_draws, 2, 2)
    source: str = ""
    extra: dict = field(default_factory=dict)   # e.g. strain-level draws

    def __post_init__(self):
        self.mu_inter = np.asarray(self.mu_inter, float)
        self.cov_inter = np.asarray(self.cov_inter, float)
        self.cov_intra = np.asarray(self.cov_intra, float)
        n = self.mu_inter.shape[0]
        assert self.mu_inter.shape == (n, 2)
        assert self.cov_inter.shape == (n, 2, 2)
        assert self.cov_intra.shape == (n, 2, 2)

    # ------------------------------------------------------------------ io
    @property
    def n_draws(self) -> int:
        return self.mu_inter.shape[0]

    def as_dict(self) -> dict:
        return dict(mu_inter=self.mu_inter, cov_inter=self.cov_inter,
                    cov_intra=self.cov_intra)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, source=np.array(self.source),
                            **self.as_dict(),
                            **{f"extra__{k}": v for k, v in self.extra.items()})

    @classmethod
    def load(cls, path: str | Path) -> "Posterior":
        z = np.load(path, allow_pickle=False)
        extra = {k[len("extra__"):]: z[k] for k in z.files if k.startswith("extra__")}
        return cls(z["mu_inter"], z["cov_inter"], z["cov_intra"],
                   source=str(z["source"]), extra=extra)

    @classmethod
    def from_cmdstan(cls, fit, source: str = "CmdStan") -> "Posterior":
        """From a cmdstanpy CmdStanMCMC object fitted with multilevel_weibull.stan."""
        mu = fit.stan_variable("mu_inter")
        ci = fit.stan_variable("cov_inter")
        cw = fit.stan_variable("cov_intra")
        extra = {}
        # strain- and curve-level estimates, used for Fig. 1 (present-day variability)
        for name in ("theta_strain", "theta_curve", "logC0"):   # (n, S, 2), (n, J, 2), (n, J)
            try:
                extra[name] = fit.stan_variable(name)
            except Exception:
                pass
        return cls(mu, ci, cw, source=source, extra=extra)

    @classmethod
    def from_pystan_pickle(cls, path: str | Path) -> "Posterior":
        """From 'Samples extracted modfied Weibull multi-level MVN stan.pkl'.

        The pickle is the dict returned by PyStan 2's fit.extract(permuted=True),
        so it holds plain numpy arrays and does not need PyStan to be read.
        """
        with open(path, "rb") as f:
            s = pickle.load(f)
        return cls(np.asarray(s["Paras0_between"]),
                   np.asarray(s["cov_between"]),
                   np.asarray(s["cov_within"]),
                   source=f"PyStan 2.19 pickle: {Path(path).name}")

    # ------------------------------------------------------------- summaries
    @staticmethod
    def _sd_corr(cov: np.ndarray):
        sd = np.sqrt(np.stack([cov[:, 0, 0], cov[:, 1, 1]], axis=1))
        corr = cov[:, 0, 1] / (sd[:, 0] * sd[:, 1])
        return sd, corr

    def summary(self) -> "pd.DataFrame":
        import pandas as pd
        sd_b, r_b = self._sd_corr(self.cov_inter)
        sd_w, r_w = self._sd_corr(self.cov_intra)
        rows = {
            "ln delta_mean": self.mu_inter[:, 0],
            "ln p_mean": self.mu_inter[:, 1],
            "SD inter ln delta": sd_b[:, 0],
            "SD inter ln p": sd_b[:, 1],
            "corr inter": r_b,
            "SD intra ln delta": sd_w[:, 0],
            "SD intra ln p": sd_w[:, 1],
            "corr intra": r_w,
        }
        out = []
        for k, v in rows.items():
            q = np.quantile(v, [0.025, 0.5, 0.975])
            out.append(dict(quantity=k, mean=v.mean(), q2_5=q[0], median=q[1], q97_5=q[2]))
        return pd.DataFrame(out).set_index("quantity")

    def delta_mean(self) -> float:
        """exp(mean of ln delta_mean): the delta used to define 2/4/6-log heating times."""
        return float(np.exp(self.mu_inter[:, 0].mean()))

    def central_draw(self) -> int:
        """Draw whose ln delta_mean is closest to the posterior mean (used to fix the target)."""
        return int(np.argmin(np.abs(self.mu_inter[:, 0] - self.mu_inter[:, 0].mean())))
