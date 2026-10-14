"""Builds the three analysis notebooks from the cell lists below.

Edit this file (or the notebooks directly) and run
    python notebooks/_build_notebooks.py
to regenerate them. Outputs are not stored here; run the notebooks to create them.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


COMMON_SETUP = r'''
import sys, os, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import campyevo
from campyevo import n, QUICK, DATA, RESULTS, FIGURES, TABLES, SIMS, POSTERIOR
from campyevo.plotting import setup_style, savefig
setup_style()
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
warnings.simplefilter("ignore", RuntimeWarning)
print("project root:", ROOT, "| QUICK mode:", QUICK)
'''

# ===========================================================================
# 01  Fit
# ===========================================================================
nb1 = [
    md(r'''
# 01 — Multilevel model of inter- and intra-strain variability (CmdStan)

Fits Equations 1a–1d of the manuscript to the plate counts of the 50 *C. jejuni*
strains heated at 55 °C (three replicate experiments per strain), and saves the
posterior draws that the simulations in notebooks 02 and 03 use.

| Level | What it describes | Output used by the simulation |
|---|---|---|
| Plate | Poisson scatter of colony counts around the true concentration (Eq. 1a) | — |
| Replicate (curve) | Replicate experiments of one strain differ: **intra-strain variability** (Eq. 1c) | `COV_intra` |
| Strain | Strains differ: **inter-strain variability** (Eq. 1d) | `mu_inter`, `COV_inter` |

**Relation to the original PyStan 2.19 fit** (notebook `...All_strain_MVN-Copy1.ipynb`, 2025)

The original fit had the same three levels (one (δ, p) per replicate curve, Eq. 1c; strain
means drawn from the inter-strain distribution, Eq. 1d). It additionally had (i) a second,
weakly identified layer around each strain mean (`within0`), (ii) the constraint that each
inter-strain SD exceeds the corresponding intra-strain SD, and (iii) narrower half-Cauchy priors
(0.1 intra, 0.5 inter). It sampled with 5.6 % divergent transitions and E-BFMI ≈ 0.1.
The model here is Eq. 1a–1d as written in the manuscript, without (i)–(iii); fitted to the same
data it reproduces the original estimates (section 6) and samples without divergences.
'''),
    code(COMMON_SETUP),
    md(r'''
## 1. Settings

* `DATA_FILE`: the plate-count table (columns `Strain_Name, rep_ID, t_heat, n_dilu, V, Nt_count`),
  found in `data/`. If it is missing, the synthetic example table is used **for testing only**.
* `CMDSTAN_PATH`: leave `None` if CmdStan was installed with conda or `install_cmdstan()`.
* `OLD_PYSTAN_PICKLE`: optional, the posterior of the original fit, for comparison.
'''),
    code(r'''
from campyevo.data import find_dataset
DATA_FILE = find_dataset(DATA)        # the plate-count table shipped in data/
EXCLUDED = ("RIMD 0366027", "RIMD 0366043", "IFTK 344")   # removed in the original notebook
CMDSTAN_PATH = None
OLD_PYSTAN_PICKLE = DATA / "Samples extracted modfied Weibull multi-level MVN stan.pkl"

CHAINS = 4
ITER_WARMUP = n(2000, 500)
ITER_SAMPLING = n(2500, 500)      # 4 x 2,500 = 10,000 posterior draws
SEED = 12345

import cmdstanpy
if CMDSTAN_PATH:
    cmdstanpy.set_cmdstan_path(str(CMDSTAN_PATH))
print("CmdStan:", cmdstanpy.cmdstan_path(), "| cmdstanpy", cmdstanpy.__version__)
'''),
    md("## 2. Data"),
    code(r'''
from campyevo.data import load_dataset, prepare

USING_SYNTHETIC = DATA_FILE is None
if USING_SYNTHETIC:
    print("!! data file not found -> using the SYNTHETIC example. Results are NOT the study's results.")
    cd = prepare(pd.read_csv(DATA / "synthetic_example.csv"), excluded=())
else:
    cd = load_dataset(DATA_FILE, excluded=EXCLUDED)

print(f"strains: {cd.n_strains}   curves (strain x replicate): {cd.n_curves}   plate counts: {len(cd.table)}")
reps = cd.summary()
print("replicates per strain:", reps["replicates"].value_counts().to_dict())
print("heating times (min):", sorted(cd.table["t_heat"].unique()))
reps.head()
'''),
    md(r'''
The counts above are what Reviewer #1 asked to see stated in the Methods
(number of strains, replicate experiments per strain, sampling times, plates per time point).
'''),
    code(r'''
reps.to_csv(TABLES / "data_structure_per_strain.csv")
'''),
    md("## 3. Fit"),
    code(r'''
from campyevo import fitting
model = fitting.compile_model()
fit = fitting.fit(cd.stan_data(), model=model, chains=CHAINS, iter_warmup=ITER_WARMUP,
                  iter_sampling=ITER_SAMPLING, seed=SEED, inits=cd.inits(),
                  output_dir=POSTERIOR / "cmdstan_csv")
'''),
    code(r'''
diag = fitting.diagnostics(fit)
print({k: v for k, v in diag.items() if k != "key_summary"})
diag["key_summary"]
'''),
    md(r'''
Check before going on: no divergent transitions, R-hat < 1.01 for every quantity,
bulk ESS of the key quantities in the thousands.
'''),
    md("## 4. Posterior summary (table for the manuscript)"),
    code(r'''
from campyevo.posterior import Posterior
post = Posterior.from_cmdstan(fit, source=("SYNTHETIC " if USING_SYNTHETIC else "") + "CmdStan " + cmdstanpy.cmdstan_version().__str__())
tab = post.summary()
tab.to_csv(TABLES / "TableS1b_posterior_summary.csv")
tab
'''),
    code(r'''
print(f"delta_mean = exp(mean ln delta_mean) = {post.delta_mean():.3f} min")
for R in (2, 4, 6):
    print(f"  {R}-log heating time = {R} x delta_mean = {R * post.delta_mean():.3f} min")
'''),
    code(r'''
post.save(POSTERIOR / ("posterior_synthetic.npz" if USING_SYNTHETIC else "posterior_cmdstan.npz"))
# Table S1a: the 50 strains, their data and their estimated parameters
tabS1a = (fitting.strain_table(fit, cd.strains)
          .merge(cd.summary().reset_index().rename(columns={"Strain_Name": "strain"}), on="strain", how="left"))
tabS1a.to_csv(TABLES / "TableS1a_strains.csv", index=False)
'''),
    md(r'''
## 5. Figure 1 — present-day variability, from the data

(a) Measured survival curves of the least and the most heat-resistant strain: points are plate
counts, thin lines the fit to each replicate experiment, thick lines the strain. δ is shown as
it is defined: the time to a 6-log reduction divided by 6 (equal to the D-value when p = 1).
(b) All 50 strains on the δ–p plane with the range of present-day strains (90 %), and the three
replicate experiments of three strains: replicates of one strain lie closer together than strains
do. This within-strain spread is what the simulation uses for the change between generations.
'''),
    code(r'''
from campyevo import plotting as P
P.setup_style_v2()
fig = P.fig1_present_day_variability(cd, post)
savefig(fig, "Fig1_present_day_variability"); plt.show()
'''),
    md(r'''
## 6. Comparison with the original PyStan posterior (optional)

If the pickle of the original fit is in `data/`, its species-level quantities are
shown next to the new fit. With the same data file the two agree closely.
'''),
    code(r'''
if OLD_PYSTAN_PICKLE.exists():
    old = Posterior.from_pystan_pickle(OLD_PYSTAN_PICKLE)
    cmp = pd.concat({"new (CmdStan)": post.summary()[["median", "q2_5", "q97_5"]],
                     "original (PyStan 2.19)": old.summary()[["median", "q2_5", "q97_5"]]}, axis=1)
    cmp.to_csv(TABLES / "posterior_new_vs_original.csv")
    display(cmp)
else:
    print("original pickle not found:", OLD_PYSTAN_PICKLE)
'''),
]

# ===========================================================================
# 02  Main simulations
# ===========================================================================
nb2 = [
    md(r'''
# 02 — Repeated heating–regrowth simulations (Figures 3–6, Figs. S2 and S8)

Each chain follows one lineage through 100 cycles. In every cycle the carried-over
cell regrows to *N*<sub>cell</sub> = 10<sup>4</sup> cells; each daughter may shift its
(ln δ, ln p) within the intra-strain variability, subject to the inter-strain
variability (Metropolis–Hastings step); the population is heated, and one survivor
starts the next cycle. 10,000 chains per condition; each chain uses its own
posterior draw, so the uncertainty of the fitted distributions is carried through.

Conditions: no heating, and heating for 2δ, 4δ and 6δ (2-, 4- and 6-log reduction
for the mean strain); start from ATCC 33560 or from the inter-strain distribution.
'''),
    code(COMMON_SETUP),
    code(r'''
from campyevo.posterior import Posterior
from campyevo.sim import SimSettings, simulate, ATCC33560
from campyevo.analysis import condition_table
from campyevo import plotting as P

# Which posterior: the new CmdStan fit, the original PyStan pickle, or the synthetic test fit.
POSTERIOR_FILE = POSTERIOR / "posterior_cmdstan.npz"
OLD_PYSTAN_PICKLE = DATA / "Samples extracted modfied Weibull multi-level MVN stan.pkl"
USE_ORIGINAL_PICKLE = False       # True = reproduce the published runs

if USE_ORIGINAL_PICKLE:
    post = Posterior.from_pystan_pickle(OLD_PYSTAN_PICKLE); tag = "pystan"
elif POSTERIOR_FILE.exists():
    post = Posterior.load(POSTERIOR_FILE); tag = "cmdstan"
else:
    post = Posterior.load(POSTERIOR / "posterior_synthetic.npz"); tag = "synthetic"
    print("!! using the SYNTHETIC posterior: for checking the code only")
print(post.source, "| draws:", post.n_draws)

N_CHAINS = n(10_000, 400)
N_CYCLES = n(100, 40)
N_CELL = 10 ** 4
SEED = 2026
d_mean = post.delta_mean()
T = {"none": 0.0, "2log": 2 * d_mean, "4log": 4 * d_mean, "6log": 6 * d_mean}
print({k: round(v, 3) for k, v in T.items()})
'''),
    code(r'''
results = {}
for init in ("atcc", "inter"):
    for c, t in T.items():
        st = SimSettings(t_heating=t, n_cycles=N_CYCLES, n_cell=N_CELL,
                         init="fixed" if init == "atcc" else "inter",
                         init_logpara=ATCC33560 if init == "atcc" else None)
        results[(init, c)] = simulate(post, st, n_chains=N_CHAINS, base_seed=SEED,
                                      cache=SIMS / f"{tag}_main_{init}_{c}_{N_CHAINS}.npz")
        print(init, c, "done")
'''),
    md("## Convergence table (replaces the eyeballed cycle counts in the Results)"),
    code(r'''
tab = condition_table({f"{i} | {c}": r for (i, c), r in results.items()})
tab.to_csv(TABLES / "main_conditions.csv", index=False)
tab
'''),
    md(r'''
*cycles to 95 %*: first cycle from which the mean ln δ stays beyond 95 % of its total
shift (plateau = mean of the last 20 cycles). Empty when there is no shift (no heating
from the inter-strain distribution).
'''),
    md(r'''
## Figures 3–6 and Figs. S2, S8

The figures follow one line of argument, each in microbiological units
(δ = time to a 6-log reduction ÷ 6, in min; p = curve shape, > 1 shoulder, < 1 tail):

| Figure | Message |
|---|---|
| 3 | One lineage, followed cycle by cycle, drifts toward greater heat resistance. |
| 4 | Across all lineages: the stronger the heating, the further they move, but the more lineages are lost. |
| 5 | Seen as survival curves: the same heating at 55 °C kills the evolved lineages less. |
| 6 | In practice: how much a treatment loses its effect, how fast, and how many lineages survive. |
'''),
    code(r'''
P.setup_style_v2()
inter = {c: results[("inter", c)] for c in T}
atcc = {c: results[("atcc", c)] for c in T}
'''),
    md("### Figure 3 — one lineage, cycle by cycle (start: ATCC 33560; the example is the lineage closest to the median end point)"),
    code(r'''
fig = P.fig3_example_lineages(atcc, post)
savefig(fig, "Fig3_example_lineage"); plt.show()
'''),
    md("### Figure 4 — where the lineages end after 100 cycles (start: present-day strains)"),
    code(r'''
fig = P.fig4_where_lineages_end(inter, post)
savefig(fig, "Fig4_where_lineages_end"); plt.show()
'''),
    md("### Figure 5 — survival curves before and after 100 cycles"),
    code(r'''
fig = P.fig5_survival_curves_before_after(inter, post, T)
savefig(fig, "Fig5_survival_curves"); plt.show()
'''),
    md("### Figure 6 — effect of the same treatment over cycles, and lineages surviving"),
    code(r'''
fig = P.fig6_treatment_effect(inter, atcc, T)
savefig(fig, "Fig6_treatment_effect"); plt.show()
'''),
    md(r'''
### Numbers for the Results text

log10 reduction achieved by each cycling treatment on the lineages still alive (median over
lineages), and the share of lineages not lost.
'''),
    code(r'''
from campyevo.analysis import reduction_over_cycles
rows = []
for (init, c), r in results.items():
    if c == "none":
        continue
    red = reduction_over_cycles(r, T[c]).set_index("cycle")["q50"]
    rows.append({"start": init, "heating": c, "heating time (min)": T[c],
                 **{f"reduction at cycle {k}": red.get(k, np.nan) for k in (0, 10, 20, 50, N_CYCLES)},
                 "lineages not lost (%)": 100 * r.survival_ratio()[-1]})
tab_eff = pd.DataFrame(rows)
tab_eff.to_csv(TABLES / "main_treatment_effect.csv", index=False)
tab_eff
'''),
    md("### Fig. S2 — mean δ and p over cycles (former Fig. 5)"),
    code(r'''
fig = P.figS2_means_over_cycles(inter, atcc)
savefig(fig, "FigS2_means_over_cycles"); plt.show()
'''),
    md("### Fig. S8 — Figures 4 and 5 for lineages started from ATCC 33560"),
    code(r'''
fig = P.fig4_where_lineages_end(atcc, post)
savefig(fig, "FigS8a_where_lineages_end_ATCC33560"); plt.show()
fig = P.fig5_survival_curves_before_after(atcc, post, T, reference=ATCC33560)
savefig(fig, "FigS8b_survival_curves_ATCC33560"); plt.show()
'''),
]

# ===========================================================================
# 03  Supplementary analyses
# ===========================================================================
nb3 = [
    md(r'''
# 03 — Supplementary analyses

| Section | Analysis | Supplementary item | Reviewer point |
|---|---|---|---|
| A | Algorithm validation: no heating → MVN<sub>inter</sub> | Fig. S1 | R2: procedures explained and justified |
| B | Exponent *L* in the acceptance step | Table S2 | R2: assumptions justified |
| C | Intra- and inter-strain SD scaling | Fig. S3, Table S3a–b | R1 p. 20: starting data set |
| D | Carry-over *n* | Fig. S4, Table S3c | R2: assumptions justified |
| **E** | **Population size after regrowth, *N*<sub>cell</sub>** (new) | **Fig. S5, Table S4** | **R1 p. 30: why exactly 4 log?** |
| F | Gain after 10 cycles vs Berdejo et al. (2024), incl. the same design (all survivors carried over) | Table S5 | R1: how the data support repeated heating |
| G | QMRA illustration | Fig. S6 | — |

Table S1 (posterior) comes from notebook 01; Figs. S2 and S8 from notebook 02; Fig. S7 is the model DAG
(former Fig. 2b); Table S6 lists the assumptions.

Item numbers follow the current draft; renumber as needed.
'''),
    code(COMMON_SETUP),
    code(r'''
from campyevo.posterior import Posterior
from campyevo.sim import SimSettings, SimResult, simulate, ATCC33560
from campyevo import analysis as A
from campyevo import plotting as P

POSTERIOR_FILE = POSTERIOR / "posterior_cmdstan.npz"
if POSTERIOR_FILE.exists():
    post = Posterior.load(POSTERIOR_FILE); tag = "cmdstan"
else:
    post = Posterior.load(POSTERIOR / "posterior_synthetic.npz"); tag = "synthetic"
    print("!! using the SYNTHETIC posterior: for checking the code only")

N_CYCLES = n(100, 40)
N_SENS = n(2000, 200)         # chains per sensitivity condition
SEED = 2026
d_mean = post.delta_mean()
T = {"none": 0.0, "2log": 2 * d_mean, "4log": 4 * d_mean, "6log": 6 * d_mean}
HEATED = ("2log", "4log", "6log")
MIN_ALIVE = 30    # plateau values are left blank when fewer lineages than this survive

def run(key, n_chains=None, **kw):
    n_chains = n_chains or N_SENS
    st = SimSettings(n_cycles=kw.pop("n_cycles", N_CYCLES), **kw)
    return simulate(post, st, n_chains=n_chains, base_seed=SEED,
                    cache=SIMS / f"{tag}_{key}_{n_chains}.npz", progress=False)
'''),
    md(r'''
## A. Algorithm validation (Fig. S1)

With no heating every cell survives, so picking one survivor is a single
Metropolis–Hastings step whose stationary distribution is MVN<sub>inter</sub>.
Started far away (ATCC 33560) the chain must move to MVN<sub>inter</sub> and stay there;
started from MVN<sub>inter</sub> it must not move. One posterior draw is fixed so that the
target is a single, known distribution.
'''),
    code(r'''
pid = post.central_draw()
mu_t, cov_t = post.mu_inter[pid], post.cov_inter[pid]
val_atcc = simulate(post, SimSettings(t_heating=0.0, n_cycles=n(400, 150), init="fixed",
                                      init_logpara=ATCC33560, posterior_id=pid),
                    n_chains=n(2000, 400), base_seed=SEED, cache=SIMS / f"{tag}_valid_atcc.npz")
val_inter = simulate(post, SimSettings(t_heating=0.0, n_cycles=n(400, 150), init="inter",
                                       posterior_id=pid),
                     n_chains=n(2000, 400), base_seed=SEED, cache=SIMS / f"{tag}_valid_inter.npz")
chk = A.stationary_check(val_atcc, mu_t, cov_t)
print(f"KS test of Mahalanobis^2 against chi2(2): D = {chk['ks_D']:.4f}, P = {chk['ks_p']:.3f}, n = {chk['n']}")
chk["table"]
'''),
    code(r'''
P.setup_style_v2()
fig = P.figS1_validation(val_atcc, val_inter, mu_t, cov_t, chk["md2"])
savefig(fig, "FigS1_algorithm_validation"); plt.show()
chk["table"].to_csv(TABLES / "validation_stationary.csv")
'''),
    md(r'''
## B. Exponent *L* (Table S2)

*L* = 1 is used for every reported result (standard Metropolis–Hastings). *L* = 0 removes the
inter-strain constraint; *L* > 1 tightens it. Varied only as a diagnostic.
'''),
    code(r'''
L_VALS = [0.0, 0.5, 1.0, 2.0]
res_L = {(L, c): run(f"L{L}_{c}", t_heating=T[c], L=L, n_chains=n(1000, 200))
         for L in L_VALS for c in ("none", "4log", "6log")}
rows = []
for (L, c), r in res_L.items():
    sm = r.summary()
    rows.append(dict(L=L, heating=c, **{"mean ln δ (final)": sm["lndelta_mean"].iloc[-1],
                "SD ln δ (final)": sm["lndelta_sd"].iloc[-1], "share surviving": sm["survival_ratio"].iloc[-1]}))
tabL = pd.DataFrame(rows).sort_values(["heating", "L"])
tabL.to_csv(TABLES / "TableS2_exponent_L.csv", index=False); tabL
'''),
    md(r'''
## C. Scaling the intra- and inter-strain standard deviations (Fig. S3, Table S3a–b)

Multipliers on the SDs (covariances × square). The intra-strain SD sets how far a
lineage can move per cycle; the inter-strain SD sets how far it can move in total.
'''),
    code(r'''
S_INTRA = [0.25, 0.5, 1.0, 2.0, 4.0]
S_INTER = [0.5, 0.75, 1.0, 1.5, 2.0]
res_w = {(s, c): run(f"sintra{s}_{c}", t_heating=T[c], s_intra=s) for s in S_INTRA for c in HEATED}
res_b = {(s, c): run(f"sinter{s}_{c}", t_heating=T[c], s_inter=s) for s in S_INTER for c in HEATED}

def sens_table(res, name):
    rows = []
    for (v, c), r in res.items():
        sm = r.summary()
        alive = int(r.alive[:, -1].sum())
        ok = alive >= MIN_ALIVE          # too few surviving lineages -> plateau not estimable
        rows.append({name: v, "heating": c,
                     "mean ln δ (plateau)": A.plateau(sm) if ok else np.nan,
                     "shift ln δ": A.plateau(sm) - sm["lndelta_mean"].iloc[0] if ok else np.nan,
                     "mean ln p (plateau)": A.plateau(sm, "lnp_mean") if ok else np.nan,
                     "cycles to 95%": A.cycles_to_fraction(sm) if ok else np.nan,
                     "share surviving": sm["survival_ratio"].iloc[-1],
                     "lineages alive": alive})
    return pd.DataFrame(rows).sort_values(["heating", name])

tab_w, tab_b = sens_table(res_w, "SD multiplier (intra)"), sens_table(res_b, "SD multiplier (inter)")
tab_w.to_csv(TABLES / "TableS3a_intra_scaling.csv", index=False)
tab_b.to_csv(TABLES / "TableS3b_inter_scaling.csv", index=False)
display(tab_w); display(tab_b)
'''),
    code(r'''
fig = P.fig_sensitivity_summary(res_w, S_INTRA, "Intra-strain SD (multiplier)", fmt=lambda v: f"×{v:g}",
                                main_value=1.0, min_alive=MIN_ALIVE, title="Intra-strain SD (change per generation)")
savefig(fig, "FigS3a_intra_scaling"); plt.show()
fig = P.fig_sensitivity_summary(res_b, S_INTER, "Inter-strain SD (multiplier)", fmt=lambda v: f"×{v:g}",
                                main_value=1.0, min_alive=MIN_ALIVE, title="Inter-strain SD (range of present-day strains)")
savefig(fig, "FigS3b_inter_scaling"); plt.show()
'''),
    md(r'''
## D. Carry-over *n* (Fig. S4, Table S3c)

*n* cells start each cycle; each regrows to *N*<sub>cell</sub>/*n* cells. *n* > 1 uses the
reference engine (whole population), so it is slower.
'''),
    code(r'''
N_CARRY = [1, 10, 100, 1000]
res_n = {(k, c): run(f"ncarry{k}_{c}", t_heating=T[c], n_carry=k, n_chains=n(1000, 100))
         for k in N_CARRY for c in HEATED}
tab_n = sens_table(res_n, "carry-over n")
tab_n.to_csv(TABLES / "TableS3c_carryover.csv", index=False); display(tab_n)
fig = P.fig_sensitivity_summary(res_n, N_CARRY, "Cells carried over to the next cycle, n", fmt=lambda v: f"{v}",
                                main_value=1, min_alive=MIN_ALIVE)
savefig(fig, "FigS4_carryover"); plt.show()
'''),
    md(r'''
## E. Population size after regrowth, *N*<sub>cell</sub> (new; Fig. S5, Table S4)

Reviewer #1: *"How can the authors know that the regrowth of the population reaches exactly 4 log?"*
10<sup>4</sup> was a scenario value. Here it is varied from 10<sup>2</sup> to 10<sup>6</sup>, and in
one further scenario it is drawn afresh in every cycle as log<sub>10</sub> *N*<sub>cell</sub> ~ Normal(4, 1)
(the variant present, commented out, in the original code).

What to look for: whether the *direction* of the shift and its ordering by heating
intensity are unchanged, and how the converged level and the share of surviving
lineages depend on *N*<sub>cell</sub>.
'''),
    code(r'''
import time
NCELL = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, ("log10normal", 4.0, 1.0)]
ncell_name = lambda v: f"10^{int(np.log10(v))}" if isinstance(v, int) else "random (10^N(4,1))"
res_N = {}
for v in NCELL:
    t0 = time.time()
    for c in HEATED + ("none",):
        res_N[(v, c)] = run(f"ncell{ncell_name(v).replace('^', '')}_{c}".replace(" ", ""),
                            t_heating=T[c], n_cell=v)
    print(f"N_cell = {ncell_name(v)}: {time.time() - t0:.0f} s")
'''),
    code(r'''
rows = []
for (v, c), r in res_N.items():
    sm = r.summary()
    alive = int(r.alive[:, -1].sum()); ok = alive >= MIN_ALIVE
    rows.append({"N_cell": ncell_name(v), "heating": c,
                 "mean ln δ (cycle 0)": sm["lndelta_mean"].iloc[0],
                 "mean ln δ (plateau)": A.plateau(sm) if ok else np.nan,
                 "shift ln δ": A.plateau(sm) - sm["lndelta_mean"].iloc[0] if ok else np.nan,
                 "mean ln p (plateau)": A.plateau(sm, "lnp_mean") if ok else np.nan,
                 "cycles to 95%": A.cycles_to_fraction(sm) if ok else np.nan,
                 "share surviving": sm["survival_ratio"].iloc[-1],
                 "lineages alive": alive})
order = {ncell_name(v): i for i, v in enumerate(NCELL)}
tab_N = pd.DataFrame(rows)
tab_N = tab_N.assign(_o=tab_N["N_cell"].map(order)).sort_values(["heating", "_o"]).drop(columns="_o")
tab_N.to_csv(TABLES / "TableS4_Ncell_sensitivity.csv", index=False)
tab_N
'''),
    code(r'''
fig = P.fig_sensitivity_summary(res_N, NCELL, "Cells after regrowth, N_cell",
                                fmt=lambda v: ncell_name(v).replace("random (10^N(4,1))", "random").replace("10^", "1e"),
                                conds=("none", "2log", "4log", "6log"), main_value=10 ** 4, min_alive=MIN_ALIVE)
savefig(fig, "FigS5_Ncell_sensitivity"); plt.show()
'''),
    md(r'''
## F. Comparison with the directed-evolution experiment of Berdejo et al. (2024) (Table S5)

**What Berdejo et al. measured.** *S.* Typhimurium LT2 was heated at 55 °C for 20 min and regrown,
ten times. The parental strain survived the treatment at 0.006 % (a 4.22-log reduction); the six
evolved lineages survived at 13–80 % (0.32 log on average). The gain after ten cycles is therefore
**3.9 log**, measured with the selection treatment itself. Control lineages regrown without heating
did not change.

**Same quantity from the simulation.** For every lineage, its ancestor (cycle 0) and its descendant
after ten cycles are given the same treatment; the gain is the difference in log reduction.

* **Matched selection strength**: the heating time is set so that the starting strain loses
  4.22 log per cycle, as LT2 did (ATCC 33560, or the mean strain for starts from the inter-strain
  distribution).
* **Matched carry-over (F2, below)**: Berdejo et al. diluted the whole heated sample into fresh
  broth, so every survivor was carried over. Section F2 simulates exactly this design. In F1,
  runs with *n* = 10 and 100 carried-over cells are kept as intermediate cases.
* For reference, the same gain is also computed for the 2-, 4- and 6-log conditions of the main
  simulations, each tested with its own cycling treatment.

For *n* > 1 the gain is computed from the mean parameters of the carried-over cells (an
approximation to the survival of a mixed population).
'''),
    code(r'''
from campyevo.sim import time_for_reduction
R_B = -np.log10(0.006 / 100)                      # LT2 parental strain: 4.22-log reduction
p_mean = float(np.exp(post.mu_inter[:, 1].mean()))
t_atcc = float(time_for_reduction(R_B, 1.80, 1.17))
t_mean = float(time_for_reduction(R_B, d_mean, p_mean))
print(f"heating time giving {R_B:.2f} log: ATCC 33560 {t_atcc:.2f} min, mean strain {t_mean:.2f} min")

runs = [  # label, start, t, n_carry, N_cell, chains
    ("matched 4.2-log", "atcc", t_atcc, 1, 10**4, n(4000, 400)),
    ("matched 4.2-log", "atcc", t_atcc, 1, 10**6, n(4000, 400)),
    ("matched 4.2-log", "inter", t_mean, 1, 10**4, n(4000, 400)),
    ("matched 4.2-log", "inter", t_mean, 1, 10**6, n(4000, 400)),
    ("matched 4.2-log", "atcc", t_atcc, 10, 10**4, n(1000, 100)),
    ("matched 4.2-log", "atcc", t_atcc, 100, 10**4, n(1000, 100)),
    ("matched 4.2-log", "atcc", t_atcc, 100, 10**6, n(200, 20)),
] + [(f"{c} (main)", i, k * d_mean, 1, 10**4, n(4000, 400))
     for c, k in [("2log", 2), ("4log", 4), ("6log", 6)] for i in ("atcc", "inter")]

rows = []
for lab, i, t, nc, N, ch in runs:
    st = SimSettings(t_heating=t, n_cycles=10, n_cell=N, n_carry=nc,
                     init="fixed" if i == "atcc" else "inter",
                     init_logpara=ATCC33560 if i == "atcc" else None)
    r = simulate(post, st, n_chains=ch, base_seed=SEED, progress=False,
                 cache=SIMS / f"{tag}_berdejo_{lab.split()[0]}_{i}_n{nc}_N{N}_{ch}.npz")
    g = A.gain_same_treatment(r, t, cycle=10)
    rows.append({"scenario": lab, "start": i, "carry-over n": nc, "N_cell": f"10^{int(np.log10(N))}",
                 "heating time (min)": t, "reduction of ancestor (log)": g["R0_median"],
                 "reduction after 10 cycles (log)": g["Rk_median"],
                 "gain, median (log)": g["gain_median"], "gain, IQR": f'{g["gain_q25"]:.2f}–{g["gain_q75"]:.2f}',
                 "share of lineages alive": g["share_alive"], "lineages": g["n_alive"]})
tab_B = pd.DataFrame(rows)
print("Berdejo et al. (2024): gain 3.90 log after 10 cycles (4.22 -> 0.32 log)")
tab_B
'''),
    md(r'''
### F2. Same design as the experiment: every survivor carried over

Design of Berdejo et al., reproduced step by step (`campyevo/benchmark.py`):

1. In every cycle, about 10<sup>8</sup> cells are heated. This assumes 50 µL of a stationary-phase culture
   at 2 × 10<sup>9</sup> CFU/mL (the cell density is an assumption; the experiment heated 50 µL).
2. The heating time gives a 4.22-log reduction of the starting strain.
3. All survivors are carried over and regrow to 10<sup>8</sup> cells, each survivor contributing the same
   number of daughter cells (for memory, at most 100,000 survivors, chosen at random, are kept as parents).
4. After 10 cycles, one surviving cell is picked at random, as a clone was picked in the experiment.
   Its log reduction under the same treatment is compared with that of the starting strain.

The log<sub>10</sub> survival of the whole population in each cycle is also recorded, the quantity
shown cycle by cycle in Fig. 1A of Berdejo et al.

Run time: about 2–5 min per lineage per CPU core at 10<sup>8</sup> cells (roughly 20–40 min for 2 × 50 lineages
on 12 cores). Results are cached in `results/sims/`.
'''),
    code(r'''
from campyevo.benchmark import WholePopSettings, simulate_whole_population, gain_table
N_HEATED = n(10 ** 8, 10 ** 5)     # cells heated per cycle (assumed: 50 uL at 2e9 CFU/mL)
N_LINEAGES = n(50, 4)
whole = {}
for i, t, init in [("atcc", t_atcc, ATCC33560), ("inter", t_mean, None)]:
    st = WholePopSettings(t_heating=t, n_heated=N_HEATED, n_cycles=10, init_logpara=init)
    whole[i] = simulate_whole_population(post, st, n_chains=N_LINEAGES, base_seed=SEED,
                                         cache=SIMS / f"{tag}_berdejo_whole_{i}_N{N_HEATED}_{N_LINEAGES}.pkl")
    g = gain_table(whole[i], t)
    rows.append({"scenario": "all survivors carried over (as in the experiment)", "start": i,
                 "carry-over n": "all survivors", "N_cell": f"10^{int(np.log10(N_HEATED))}",
                 "heating time (min)": t, "reduction of ancestor (log)": g["R0_median"],
                 "reduction after 10 cycles (log)": g["Rk_median"],
                 "gain, median (log)": g["gain_median"], "gain, IQR": f'{g["gain_q25"]:.2f}–{g["gain_q75"]:.2f}',
                 "share of lineages alive": g["share_alive"], "lineages": g["n_alive"]})

# check: does the assumed number of heated cells matter? (10x fewer cells, start ATCC 33560)
N_CHECK = n(10 ** 7, 10 ** 5)
st = WholePopSettings(t_heating=t_atcc, n_heated=N_CHECK, n_cycles=10, init_logpara=ATCC33560)
chk = simulate_whole_population(post, st, n_chains=N_LINEAGES, base_seed=SEED,
                                cache=SIMS / f"{tag}_berdejo_whole_atcc_N{N_CHECK}_{N_LINEAGES}.pkl")
g = gain_table(chk, t_atcc)
rows.append({"scenario": "all survivors carried over, 10x fewer cells heated (check)", "start": "atcc",
             "carry-over n": "all survivors", "N_cell": f"10^{int(np.log10(N_CHECK))}",
             "heating time (min)": t_atcc, "reduction of ancestor (log)": g["R0_median"],
             "reduction after 10 cycles (log)": g["Rk_median"],
             "gain, median (log)": g["gain_median"], "gain, IQR": f'{g["gain_q25"]:.2f}–{g["gain_q75"]:.2f}',
             "share of lineages alive": g["share_alive"], "lineages": g["n_alive"]})
tab_B = pd.DataFrame(rows)
tab_B.to_csv(TABLES / "TableS5_Berdejo_comparison.csv", index=False)

# population survival in each cycle (median over lineages), cf. Fig. 1A of Berdejo et al.
surv = pd.DataFrame({i: whole[i][[f"log_surv_cycle{c}" for c in range(1, 11)]].median().values
                     for i in whole}, index=pd.Index(range(1, 11), name="cycle"))
surv.columns = ["start: ATCC 33560", "start: present-day strains"]
surv.to_csv(TABLES / "TableS5b_population_survival_per_cycle.csv")
print("Berdejo et al. (2024): parental strain -4.22 log; evolved clones -0.32 log (mean)")
display(tab_B); surv
'''),
    md(r'''
## G. QMRA illustration (Fig. S6)

A treatment intended to give a 4-log reduction (4δ<sub>mean</sub>) applied to a portion at an
assumed 2 log CFU; Beta-Poisson dose–response with α = 0.145, β = 7.59. These are the values of
Medema et al. (1996); **the draft cites Teunis et al. (2005) for them — please check the reference.**
'''),
    code(r'''
f6 = SIMS / f"{tag}_main_inter_6log_{n(10_000, 400)}.npz"
base = A.baseline_strain_draws(post, n=n(20_000, 4000))
conv = SimResult.load(f6).final_params() if f6.exists() else run("qmra_6log", t_heating=T["6log"]).final_params()
r_base, _ = A.risk_for_params(base, T["4log"], log10_N0=2.0)
r_conv, _ = A.risk_for_params(conv, T["4log"], log10_N0=2.0)
tab_Q = pd.DataFrame({"scenario": ["present-day strains", "after 100 cycles at 6-log"],
                      "mean P(infection)": [r_base.mean(), r_conv.mean()],
                      "95th percentile": [np.quantile(r_base, .95), np.quantile(r_conv, .95)]})
tab_Q.to_csv(TABLES / "TableS_QMRA.csv", index=False); display(tab_Q)
fig = P.figS6_qmra(r_base, r_conv)
savefig(fig, "FigS6_QMRA"); plt.show()
'''),
]

# ===========================================================================
# 04  Figures -> PowerPoint and PDF
# ===========================================================================
nb4 = [
    md(r'''
# 04 — All figures as PowerPoint and PDF

Run after 01–03 (they save every figure to `results/figures/` as PDF, PNG and SVG;
02 and 03 read cached simulations, so re-running them only redraws the figures).

Output in `results/`:

| File | Content |
|---|---|
| `Figures_main.pptx`, `Figures_supplementary.pptx` | one slide per figure, at its real size; legend in the speaker notes |
| `Figures_main.pdf`, `Figures_supplementary.pdf` | the same, one figure per page (vector) |
| `figure_legends.txt` | all legends, ready to paste into the manuscript |

**Editing in PowerPoint.** Each figure is placed as an SVG (vector) with a PNG fallback.
Right-click → *Convert to Shape* (PowerPoint 365 / 2019 or later) makes every line and every
text box editable. Fonts are Arial at the sizes used in the paper.

**Figures drawn by hand.** Fig. 2 (flowchart) and Fig. S7 (model DAG) are not made by code.
Save them as `figures_manual/Fig2_one_cycle.svg` (or `.png`) and
`figures_manual/FigS7_model_DAG.svg` (or `.png`); otherwise a placeholder slide is inserted.

Legends and take-home messages are edited in `campyevo/figure_deck.py`.
'''),
    code(COMMON_SETUP),
    code(r'''
# Packages for this notebook only (python-pptx is imported as "pptx").
# Missing ones are installed with conda (conda-forge) into the environment of this kernel.
import sys, os, importlib, importlib.util
NEEDED = {"pptx": "python-pptx", "pypdf": "pypdf", "PIL": "pillow"}
missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]
if missing:
    if os.path.isdir(os.path.join(sys.prefix, "conda-meta")):      # the kernel runs in a conda environment
        print("installing with conda:", " ".join(missing), "->", sys.prefix)
        %conda install -y -q -c conda-forge {" ".join(missing)}
        importlib.invalidate_caches()
    else:
        raise RuntimeError("this kernel is not a conda environment; in a terminal run\n"
                           f"  conda activate campyevo\n  conda install -c conda-forge {' '.join(missing)}\n"
                           "then restart the kernel")
print("ok:", {mod: importlib.util.find_spec(mod) is not None for mod in NEEDED})
'''),
    code(r'''
from campyevo import figure_deck as FD
MANUAL = ROOT / "figures_manual"
MANUAL.mkdir(exist_ok=True)
LABEL_SLIDES = True     # False: no figure number / message above the figure on each slide

rep_main = FD.build_pptx(FD.MAIN, FIGURES, RESULTS / "Figures_main.pptx", manual_dir=MANUAL,
                         label_slides=LABEL_SLIDES)
rep_sup = FD.build_pptx(FD.SUPPLEMENTARY, FIGURES, RESULTS / "Figures_supplementary.pptx",
                        manual_dir=MANUAL, label_slides=LABEL_SLIDES)
pd.DataFrame(rep_main + rep_sup)
'''),
    code(r'''
miss = FD.build_pdf(FD.MAIN, FIGURES, RESULTS / "Figures_main.pdf", manual_dir=MANUAL)
miss += FD.build_pdf(FD.SUPPLEMENTARY, FIGURES, RESULTS / "Figures_supplementary.pdf", manual_dir=MANUAL)
FD.write_legends(FD.ALL, RESULTS / "figure_legends.txt")
print("not in the PDFs (no PDF file):", miss or "none")
print((RESULTS / "figure_legends.txt").read_text()[:1500])
'''),
]

for name, cells in [("01_fit_multilevel_model", nb1), ("02_main_simulations", nb2),
                    ("03_supplementary_analyses", nb3), ("04_figures_powerpoint", nb4)]:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3 (campyevo)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, HERE / f"{name}.ipynb")
    print("wrote", name)
