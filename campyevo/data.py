"""Loading the survival-curve data set and turning it into Stan input.

The expected input is the plate-count table used for Abe & Kawasaki (2024,
2025), shipped in this repository as ``data/Cam_all_55C_2024_01_31.csv``,
with at least these columns:

    Strain_Name   strain name
    rep_ID        replicate experiment number within the strain
    t_heat        heating time (min)
    n_dilu        number of 10-fold dilutions plated
    V             plated volume
    Nt_count      colony count on the plate

A synthetic data set with the same columns can be generated with
:func:`make_synthetic_dataset` to check the pipeline before the real data are
in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["Strain_Name", "rep_ID", "t_heat", "n_dilu", "V", "Nt_count"]

# Strains removed in the original fitting notebook.
DEFAULT_EXCLUDED = ("RIMD 0366027", "RIMD 0366043", "IFTK 344")

# IFTK 33 was measured twice: three replicate experiments on the time grid used
# for every other strain (rep_ID 1-3, six times up to 11.25 min) and three on a
# finer grid (rep_ID 4-6, six times up to 5.617 min). The finer set was the one
# used, so the earlier three replicates are excluded, leaving three replicate
# experiments for each of the 50 strains (150 survival curves, 1,276 plate
# counts).
DEFAULT_EXCLUDED_REPS = {"IFTK 33": (1, 2, 3)}


@dataclass
class CurveData:
    """Plate counts indexed by curve (one strain x one replicate experiment)."""

    table: pd.DataFrame          # one row per plate count, with curve / strain indices
    curves: pd.DataFrame         # one row per curve
    strains: list[str]           # strain names, position = strain index - 1

    @property
    def n_strains(self) -> int:
        return len(self.strains)

    @property
    def n_curves(self) -> int:
        return len(self.curves)

    def stan_data(self, logC0_lower: float = 6.0, logC0_upper: float = 10.0) -> dict:
        t = self.table
        return dict(
            N=int(len(t)),
            S=int(self.n_strains),
            J=int(self.n_curves),
            strain_of_curve=self.curves["strain"].astype(int).tolist(),
            curve=t["curve"].astype(int).tolist(),
            t_heat=t["t_heat"].astype(float).tolist(),
            n_dilu=t["n_dilu"].astype(float).tolist(),
            V=t["V"].astype(float).tolist(),
            count=t["Nt_count"].astype(int).tolist(),
            logC0_lower=float(logC0_lower),
            logC0_upper=float(logC0_upper),
        )

    def curve_estimates(self, mu=(0.26, 0.0)) -> pd.DataFrame:
        """Quick per-curve least-squares fit of log10 counts (starting values only).

        y = log10(count / V) + d  is fitted with  y = log10 N_0 - 6 (t / 6 delta)^p
        using plates with at least one colony. Curves that cannot be fitted get
        the species-level starting value.
        """
        from scipy.optimize import least_squares
        rows = []
        for j, g in self.table.groupby("curve"):
            g = g[g["Nt_count"] > 0]
            y = np.log10(g["Nt_count"] / g["V"]) + g["n_dilu"]
            t = g["t_heat"].to_numpy(float)
            y = y.to_numpy(float)
            y0 = float(y[t == t.min()].mean()) if len(y) else 8.0
            x0 = np.array([y0, mu[0], mu[1]])
            ok = False
            if len(y) >= 3 and (t > 0).any():
                def res(x):
                    return y - (x[0] - 6.0 * (t / (6.0 * np.exp(x[1]))) ** np.exp(x[2]))
                try:
                    r = least_squares(res, x0, bounds=([0, -3, -1.5], [12, 3, 1.5]))
                    x0, ok = r.x, r.success
                except Exception:
                    pass
            rows.append(dict(curve=j, logC0=x0[0], lnd=x0[1], lnp=x0[2], fitted=ok))
        return pd.DataFrame(rows).set_index("curve")

    def inits(self, logC0_lower: float = 6.0, logC0_upper: float = 10.0) -> dict:
        """Starting values for NUTS built from the per-curve least-squares fits.

        With thousands of colonies per curve the posterior is very concentrated,
        and Stan's default random starting values sit so far from it that warm-up
        stalls. Curve, strain and species values are therefore started from the
        quick fits, the standard deviations from their empirical spread.
        """
        ce = self.curve_estimates()
        lo, hi = logC0_lower + 0.05, logC0_upper - 0.05
        theta_c = ce[["lnd", "lnp"]].to_numpy()
        strain_of = self.curves["strain"].to_numpy()
        theta_s = np.array([theta_c[strain_of == s].mean(axis=0) for s in range(1, self.n_strains + 1)])
        mu = theta_s.mean(axis=0)
        sd_b = np.clip(theta_s.std(axis=0), 0.05, 2.0)
        resid = theta_c - theta_s[strain_of - 1]
        sd_w = np.clip(resid.std(axis=0), 0.02, 1.0)
        return dict(
            logC0=np.clip(ce["logC0"].to_numpy(), lo, hi).tolist(),
            logdelta_mean=float(np.clip(mu[0], -4.9, 1.55)),
            logpower_mean=float(np.clip(mu[1], -0.95, 0.95)),
            sigma_inter=sd_b.tolist(), sigma_intra=sd_w.tolist(),
            Lcorr_inter=[[1.0, 0.0], [0.0, 1.0]], Lcorr_intra=[[1.0, 0.0], [0.0, 1.0]],
            theta_strain=theta_s.tolist(),
            theta_curve=theta_c.tolist(),
        )

    def summary(self) -> pd.DataFrame:
        g = self.table.groupby("Strain_Name", sort=False)
        return pd.DataFrame({
            "replicates": g["rep"].nunique(),
            "plates": g.size(),
            "time points": g["t_heat"].nunique(),
        })


# Names the plate-count table has had. The first is the one in this repository;
# the others are the names the file carries in the authors' own directories.
DATASET_NAMES = (
    "Cam_all_55C_2024_01_31.csv",
    "Cam_all_55ºC_2024_01_31_deleted.csv",
    "Cam_all_55_C_2024_01_31.csv",
)


def find_dataset(directory: str | Path) -> Path | None:
    """The plate-count table in `directory`, under any of the names it has had."""
    directory = Path(directory)
    for name in DATASET_NAMES:
        if (directory / name).exists():
            return directory / name
    return None


def load_dataset(path: str | Path,
                 excluded: tuple[str, ...] | list[str] = DEFAULT_EXCLUDED,
                 drop_na_counts: bool = True) -> CurveData:
    """Read the plate-count CSV and index it by strain and by curve."""
    # the file is written by Excel and may carry a byte-order mark
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from {path}: {missing}")
    return prepare(df, excluded=excluded, drop_na_counts=drop_na_counts)


def prepare(df: pd.DataFrame,
            excluded=DEFAULT_EXCLUDED,
            drop_na_counts: bool = True,
            excluded_reps=DEFAULT_EXCLUDED_REPS) -> CurveData:
    df = df.copy()
    if excluded:
        df = df[~df["Strain_Name"].isin(list(excluded))]
    for strain, reps in (excluded_reps or {}).items():
        df = df[~(df["Strain_Name"].eq(strain) & df["rep_ID"].isin(list(reps)))]
    if drop_na_counts:
        df = df[df["Nt_count"].notna()]
    df["Nt_count"] = df["Nt_count"].round().astype(int)
    if (df["Nt_count"] < 0).any():
        raise ValueError("negative colony counts in the data")

    # strain index in order of first appearance (as in the original notebook)
    strains = list(dict.fromkeys(df["Strain_Name"].tolist()))
    df["strain"] = df["Strain_Name"].map({s: i + 1 for i, s in enumerate(strains)})

    # replicate index renumbered 1..K within each strain
    df["rep"] = (df.groupby("strain")["rep_ID"]
                   .transform(lambda r: pd.factorize(r, sort=True)[0] + 1))

    curves = (df[["strain", "rep", "Strain_Name"]]
              .drop_duplicates()
              .sort_values(["strain", "rep"])
              .reset_index(drop=True))
    curves.index = curves.index + 1
    curves["curve"] = curves.index
    df = df.merge(curves[["strain", "rep", "curve"]], on=["strain", "rep"], how="left")

    return CurveData(table=df.reset_index(drop=True), curves=curves, strains=strains)


# ---------------------------------------------------------------------------
# Synthetic data (pipeline check / parameter recovery)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n_strains: int = 50,
                           n_reps: int = 3,
                           times=(0.0, 2.25, 4.5, 6.75, 9.0, 11.25),
                           plates_per_point: int = 2,
                           mu=(0.259, 0.005),
                           sd_inter=(0.258, 0.306),
                           corr_inter=-0.455,
                           sd_intra=(0.10, 0.12),
                           corr_intra=0.0,
                           logC0_mean: float = 8.0,
                           logC0_sd: float = 0.3,
                           V: float = 0.1,
                           seed: int = 1) -> tuple[pd.DataFrame, dict]:
    """Simulate a plate-count table from known parameters.

    Defaults for the species level are the published posterior means; the
    intra-strain values are placeholders. Returns (table, truth).
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    s_b = np.asarray(sd_inter, float)
    s_w = np.asarray(sd_intra, float)
    cov_b = np.array([[s_b[0] ** 2, corr_inter * s_b[0] * s_b[1]],
                      [corr_inter * s_b[0] * s_b[1], s_b[1] ** 2]])
    cov_w = np.array([[s_w[0] ** 2, corr_intra * s_w[0] * s_w[1]],
                      [corr_intra * s_w[0] * s_w[1], s_w[1] ** 2]])
    theta_s = rng.multivariate_normal(mu, cov_b, size=n_strains)
    rows = []
    for s in range(n_strains):
        for r in range(n_reps):
            lnd, lnp = rng.multivariate_normal(theta_s[s], cov_w)
            d, p = np.exp(lnd), np.exp(lnp)
            logC0 = rng.normal(logC0_mean, logC0_sd)
            for t in times:
                log10N = logC0 - 6.0 * (t / (6.0 * d)) ** p
                # choose the dilution that would give about 30-300 colonies
                n_dilu = max(0, int(np.floor(log10N + np.log10(V) - 2.0)))
                for _ in range(plates_per_point):
                    lam = 10 ** (log10N - n_dilu) * V
                    rows.append(dict(Strain_Name=f"SYN{s + 1:03d}", rep_ID=r + 1,
                                     t_heat=t, n_dilu=n_dilu, V=V,
                                     Nt_count=int(rng.poisson(lam))))
    truth = dict(mu=mu, cov_inter=cov_b, cov_intra=cov_w, theta_strain=theta_s)
    return pd.DataFrame(rows), truth
