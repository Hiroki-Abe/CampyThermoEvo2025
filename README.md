# CampyThermoEvo

Data, code and notebooks for

> **Simulating the selection of heat-resistant *Campylobacter jejuni* during
> repeated heating, from the variability measured among and within strains**
> Hiroki Furugori Abe, Susumu Kawasaki, Tomoya Yoshinari, Takahiro Ohnishi.
> *Microbiology Spectrum* (2026).

**What the study does.** Bacteria of the same species differ in how well they
survive heating. Risk assessment usually treats that difference as a fixed
property of the species. Here it is treated as something that can change: a
population is heated, the survivors regrow, and the cycle repeats. The
simulation is built from survival curves of 50 *C. jejuni* strains measured at
55 °C, using both the variability **between** strains and the variability
**between replicate experiments of one strain**.

Everything in the paper can be reproduced from this repository: the four
notebooks run in order produce every figure, every table and every number
quoted in the main text and in the supplementary material.

---

## What is here

```
data/
  Cam_all_55C_2024_01_31.csv     plate counts, 53 strains x 3 replicate experiments at 55 °C
  synthetic_example.csv          fake data with the same columns, to check the pipeline
campyevo/                        the Python package
  data.py        plate-count table -> Stan input; synthetic data for testing
  fitting.py     CmdStan wrappers and diagnostics
  posterior.py   the posterior draws the simulation uses
  sim.py         the simulation engine (reference and fast engines, parallel runner)
  benchmark.py   the whole-population variant used for the comparison with Berdejo et al.
  analysis.py    convergence, validation, risk-assessment and benchmark summaries
  plotting.py    the figures
  figure_deck.py the figure list, the legends, and the PowerPoint / PDF assembly
stan/
  multilevel_weibull.stan        the model of Equations 1a-1d
notebooks/                       run 01 -> 02 -> 03 -> 04
figures_manual/                  the two hand-drawn figures (Fig. 2 and Fig. S7)
results/                         output: figures, tables, cached simulations, posterior
tests/test_engine.py             checks of the simulation engine
```

### The data set

`data/Cam_all_55C_2024_01_31.csv` is the plate-count table of Abe & Kawasaki
(2024, 2025), where the experimental procedures are described. One row is one
plate count:

| column | meaning |
|---|---|
| `Strain_Name` | strain |
| `rep_ID` | replicate experiment within the strain |
| `t_heat` | heating time (min) |
| `n_dilu` | number of 10-fold dilutions plated |
| `V` | plated volume (mL) |
| `Nt_count` | colony count on the plate |

The code applies two exclusions, both made in the original analysis and both
stated in `campyevo/data.py`:

* the strains RIMD 0366027, RIMD 0366043 and IFTK 344 are dropped;
* IFTK 33 was measured twice, and the earlier three replicate experiments
  (`rep_ID` 1-3, on the coarser time grid) are dropped in favour of the later
  three.

What is left is what the paper analyses: **50 strains, 150 survival curves,
1,276 plate counts**.

In these experiments each strain was heated **once**. Cells that survived were
not regrown and heated again, so the data contain no information about repeated
heating; that is what the simulation supplies, under the assumptions listed in
Table S6 of the paper.

---

## Which notebook makes what

| Notebook | Output in the paper |
|---|---|
| `01_fit_multilevel_model.ipynb` | Fit of Equations 1a-1d with CmdStan, diagnostics, **Fig. 1**, **Table S1a**, **Table S1b** |
| `02_main_simulations.ipynb` | The 100-cycle simulations: **Figs 3, 4, 5, 6**, **Fig. S2**, **Fig. S8**, the convergence and treatment-effect tables |
| `03_supplementary_analyses.ipynb` | Validation of the algorithm (**Text S1**, **Fig. S1**, **Table S2**); the sensitivity analyses (**Figs S3a, S3b, S4, S5**, **Tables S3a, S3b, S3c, S4**); the comparison with Berdejo et al. (**Table S5**, **Table S5b**); the risk-assessment illustration (**Fig. S6**) |
| `04_figures_powerpoint.ipynb` | All figures assembled into `Figures_main.pptx` / `.pdf`, `Figures_supplementary.pptx` / `.pdf` and `figure_legends.txt` |

**Fig. 2** (one heating–regrowth cycle) and **Fig. S7** (the same cycle written
formally) are drawn by hand and live in `figures_manual/`; notebook 04 places
them in the deck.

The CSV files in `results/tables/` are the output of the authors' own run and
back every number quoted in the paper, so they can be checked without running
anything.

---

## Setup

Tested on macOS (Apple silicon) and Linux.

```bash
# 1. a C++ compiler for CmdStan (macOS)
xcode-select --install

# 2. the environment, with CmdStan from conda-forge (recommended)
conda env create -f environment.yml
conda activate campyevo

#    or, without conda:
#    python3 -m venv .venv && source .venv/bin/activate
#    pip install -r requirements.txt
#    python -c "import cmdstanpy; cmdstanpy.install_cmdstan(cores=8)"

# 3. check
python -c "import cmdstanpy; print(cmdstanpy.cmdstan_path())"
python -m pytest tests -q
```

## Running

```bash
cd notebooks
jupyter lab          # run 01 -> 02 -> 03 -> 04 in order
```

Figures are written to `results/figures/` as PDF, PNG and SVG (the SVG keeps
text editable). Notebook 04 builds the PowerPoint and PDF decks: one slide per
figure at its real size, the SVG with a PNG fallback, and the legend in the
speaker notes. In PowerPoint, right-click → *Convert to Shape* makes every
element editable.

Simulations are cached in `results/sims/`, so after the first run, changing a
figure and re-running 02 → 04 takes about a minute. Delete a cache file to
re-run that condition.

```bash
CAMPYEVO_QUICK=1 jupyter lab     # every run shrunk; finishes in minutes, code check only
```

Chain *k* draws from `default_rng([seed, k])`, so a result does not depend on
how many processes ran it.

If the data file is missing, the notebooks fall back to
`data/synthetic_example.csv` and say so in their output. Those numbers are not
the study's results.

---

## The two simulation engines

Both implement the same model. One cycle is: regrowth to *N*<sub>cell</sub>
cells with intra-strain divergence, constrained by a Metropolis–Hastings step
against the inter-strain distribution; then stochastic thermal death of every
cell; then carry-over of *n* survivors.

* `reference` draws all *N*<sub>cell</sub> cells every cycle, as the published
  code did.
* `fast` (used automatically when *n* = 1) draws cells in blocks and stops at
  the first survivor. The cells are independent and identically distributed, so
  the first survivor has the same distribution as a survivor picked at random,
  and a lineage is lost exactly when none of the *N*<sub>cell</sub> cells
  survives. The two engines therefore define the same probability model;
  `tests/test_engine.py` compares them.

## Notation

| Paper | Code |
|---|---|
| (ln δ<sub>mean</sub>, ln p<sub>mean</sub>) | `Posterior.mu_inter` |
| COV<sub>inter</sub>, COV<sub>intra</sub> | `Posterior.cov_inter`, `Posterior.cov_intra` |
| *N*<sub>cell</sub> | `SimSettings.n_cell` |
| *n* (carry-over) | `SimSettings.n_carry` |
| *L* (acceptance-step exponent) | `SimSettings.L` |
| SD multipliers | `SimSettings.s_intra`, `SimSettings.s_inter` |
| *t*<sub>heating</sub> | `SimSettings.t_heating` |

The Stan model keeps `between` and `within` in its variable names, inherited
from the original code: `cov_between` = COV<sub>inter</sub> and `cov_within` =
COV<sub>intra</sub>. The Python code uses `inter` and `intra`, as the paper
does.

Modified Weibull model: log<sub>10</sub> *N*<sub>t</sub> = log<sub>10</sub>
*N*<sub>0</sub> − 6 (*t* / 6δ)<sup>*p*</sup>. The time to a 6-log reduction is
6δ for any *p*, so δ is the mean time per log reduction over a 6-log reduction,
and δ equals the D-value when *p* = 1.

---

## Citing

See `CITATION.cff`, or cite the article above. Code is released under the MIT
licence and the data under CC BY 4.0; see `LICENSE`.

The single-notebook code of the previous version of this work is kept in the
history of this repository; the branch
[`v0-original`](https://github.com/Hiroki-Abe/CampyThermoEvo2025/tree/v0-original)
points at its last commit.

## Archiving

For a citable, permanent snapshot, link this repository to
[Zenodo](https://zenodo.org/account/settings/github/) and publish a release.
Zenodo mints a DOI for the release, which can be quoted in the Data
availability statement alongside the repository URL.
