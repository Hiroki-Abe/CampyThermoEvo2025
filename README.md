# CampyThermoEvo2025

This is the repository for the manuscript


# Notation and parameters

**A Markov chain Monte Carlo framework for exploring parameter dynamics under repeated thermal
inactivation with inter- and intra-strain variability**

This document defines every symbol used in the model and maps each one onto the corresponding
variable name in the code. Throughout, the subscripts **inter** and **intra** are used
consistently; the variable names in the code use `between` and `within` for the same two levels.

---

## 1. Inactivation model

Individual-cell survival is described by a modified Weibull model:

$$\log_{10} N_t \;=\; \log_{10} N_0 \;-\; 6\left(\frac{t}{6\delta}\right)^{p}$$

| Symbol | Definition | Unit |
|---|---|---|
| $\delta$ | Characteristic time per log-reduction, averaged over a 6-log reduction | min |
| $p$ | Shape parameter of the modified Weibull model | – |
| $N_0,\;N_t$ | Viable count before heating and after heating for time $t$ | CFU mL⁻¹ |
| $t$ | Heating time | min |

### The meaning of δ

The time required to reach a 6-log reduction is **6δ**, not δ, and this holds **for any value
of p**:

$$t\;=\;6\delta \;\;\Longrightarrow\;\; 6\left(\frac{6\delta}{6\delta}\right)^{p} = 6 \text{ log},
\qquad \text{for all } p$$

Equivalently, writing $\delta_6$ for the 6-log reduction time,

$$\delta \;=\; \frac{\delta_6}{6}$$

so δ is the **mean characteristic time per one log-reduction**, averaged over a 6-log reduction.

### Why δ is rescaled

The classical Mafart δ is the time for the **first** decimal reduction. When a survival curve has
a shoulder ($p>1$) or a tail ($p<1$), that first log is not representative of the inactivation
behavior as a whole. Five hypothetical strains with an identical classical δ of 1.00 min differ
six-fold in their actual 6-log times:

| $p$ | Curve shape | Classical δ (1-log time) | Actual 6-log time |
|---|---|---|---|
| 0.6 | strong tailing | 1.00 min | 19.81 min |
| 0.7 | tailing | 1.00 min | 12.93 min |
| 1.0 | log-linear | 1.00 min | 6.00 min |
| 1.3 | shoulder | 1.00 min | 3.97 min |
| 1.5 | strong shoulder | 1.00 min | 3.30 min |

A parameter that assigns the same value to strains needing 3.3 min and 19.8 min for the same
practical outcome cannot rank strains by heat resistance. The rescaling also removes an
entanglement between the two parameters that would otherwise make the covariance structure
uninterpretable: sampling from the fitted inter-strain distribution gives
corr(ln δ, ln p) = **−0.00** under the parameterization used here, against **−0.78** under the
classical parameterization. Because the method is a random walk *in the* (δ, p) *plane*, a
shape-independent δ is a prerequisite rather than a convenience.

The two parameterizations are related by

$$\delta \;=\; \delta_{\text{classical}}\cdot \frac{6^{1/p}}{6}$$

### Generalization to other reduction targets

The time to reach any target $R$ (in log₁₀) is

$$t_R \;=\; 6\delta\left(\frac{R}{6}\right)^{1/p}$$

which returns $t = 6\delta$ at $R = 6$ for every $p$. For the fitted parameters
(δ = 1.296 min, p = 1.024), the three heating durations used in this study are:

| Label | Heating time | Achieved reduction (95% CrI) |
|---|---|---|
| 2-log | 2.591 min ( = 2δ ) | 1.945 log (1.773–2.126) |
| 4-log | 5.182 min ( = 4δ ) | 3.959 log (3.826–4.091) |
| 6-log | 7.774 min ( = 6δ ) | 6.000 log (exact for any $p$) |

At $p = 1$ the durations $2\delta$, $4\delta$ and $6\delta$ give exactly 2-, 4- and 6-log
reductions; the small departures above reflect the fitted $p$ of 1.024.

---

## 2. Observation model

Colony counts are modeled as Poisson, so that measurement noise from plate counting is separated
from biological variability during Bayesian inference:

$$N_{\text{plate},t,l,k} \sim \text{Poisson}\!\left(10^{\log N_{t,l,k}} \cdot \frac{V_{\text{plate}}}{10^{d}}\right)$$

| Symbol | Definition |
|---|---|
| $N_{\text{plate},t,l,k}$ | Colony count on a plate for strain $l$, heating time $t$, replicate $k$ |
| $V_{\text{plate}}$ | Plated volume |
| $d$ | Number of 10-fold dilutions |
| $l$ | Strain index, $l = 1,\dots,n_{\text{strain}}$ (50 strains) |
| $k$ | Replicate index, $k = 1,\dots,n_{\text{replicates}}$ |

---

## 3. Hierarchical variability

Both levels of variability act on the **logarithms** of the two parameters, which keeps both
parameters positive and makes the joint distribution approximately normal.

$$\begin{pmatrix}\ln\delta_{l,k}\\ \ln p_{l,k}\end{pmatrix}
\sim \text{MVN}_{\text{intra}}\!\left[\begin{pmatrix}\ln\delta_{l,\text{mean}}\\ \ln p_{l,\text{mean}}\end{pmatrix},\;\text{COV}_{\text{intra}}\right]$$

$$\begin{pmatrix}\ln\delta_{l,\text{mean}}\\ \ln p_{l,\text{mean}}\end{pmatrix}
\sim \text{MVN}_{\text{inter}}\!\left[\begin{pmatrix}\ln\delta_{\text{mean}}\\ \ln p_{\text{mean}}\end{pmatrix},\;\text{COV}_{\text{inter}}\right]$$

| Symbol | Definition |
|---|---|
| $\text{MVN}_{\text{inter}}$ | Multivariate normal distribution of $(\ln\delta,\ln p)$ **among** strains |
| $\text{MVN}_{\text{intra}}$ | Multivariate normal distribution of $(\ln\delta,\ln p)$ **within** a strain |
| $\text{COV}_{\text{inter}}$ | Covariance matrix of $\text{MVN}_{\text{inter}}$ |
| $\text{COV}_{\text{intra}}$ | Covariance matrix of $\text{MVN}_{\text{intra}}$ |
| $\text{COV}_{\text{chol,inter}}$, $\text{COV}_{\text{chol,intra}}$ | Cholesky factors used during Stan fitting; $\text{COV} = \text{COV}_{\text{chol}}\cdot\text{COV}_{\text{chol}}^{\mathsf T}$ |

Fitting used the `MultiNormalCholesky` parameterization, which is mathematically equivalent to
the multivariate normal form above and improves sampling efficiency.

**Interpretation used throughout:** inter-strain variability acts as the *gravity of the species*,
bounding where a lineage can go; intra-strain variability acts as the *step size*, determining how
far it can move per cycle.

Posterior estimates (mean of the posterior draws):

| Quantity | Value |
|---|---|
| $\ln\delta_{\text{mean}}$ | 0.259 (δ = 1.296 min) |
| $\ln p_{\text{mean}}$ | 0.005 (p = 1.024) |
| SD of $\ln\delta$ across strains | 0.258 |
| SD of $\ln p$ across strains | 0.306 |
| corr($\ln\delta$, $\ln p$) across strains | −0.455 |

---

## 4. Metropolis–Hastings acceptance step

Each cell proposes a new parameter couple from $\text{MVN}_{\text{intra}}$ centered on its parent,
and the proposal is accepted with probability

$$P(\text{accept}) \;=\; \min\!\left(1,\; LR^{\,L}\right),
\qquad
LR \;=\;
\frac{f_{\text{MVN}_{\text{inter}}}\!\left[\left(\begin{smallmatrix}\ln\delta\\ \ln p\end{smallmatrix}\right)_{\text{candidate}}\right]}
     {f_{\text{MVN}_{\text{inter}}}\!\left[\left(\begin{smallmatrix}\ln\delta\\ \ln p\end{smallmatrix}\right)_{\text{prior}}\right]}$$

| Symbol | Definition |
|---|---|
| $f_{\text{MVN}}$ | **Probability density function** of a multivariate normal distribution |
| $LR$ | Ratio of the $\text{MVN}_{\text{inter}}$ densities of the candidate and the current couple |
| $L$ | Exponent applied to $LR$ in the acceptance step |

> **Note on $f_{\text{MVN}}$.** Earlier drafts wrote this as $L_{\text{MVN}}$ ("likelihood"), which
> was ambiguous: the probability of any single point under a continuous distribution is zero.
> $LR$ is a ratio of probability **densities**, and the notation $f_{\text{MVN}}$ is used here to
> make that explicit.

### The role of $L$

Because the proposal distribution $\text{MVN}_{\text{intra}}$ is symmetric about the current value,
the step above is a **random-walk Metropolis algorithm**, and at $L = 1$ its stationary
distribution is exactly $\text{MVN}_{\text{inter}}$.

| $L$ | Behavior | Stationary distribution |
|---|---|---|
| 0 | No species-level constraint; free diffusion | none (unbounded) |
| **1** | **Standard Metropolis–Hastings — used for every result in this study** | $\text{MVN}_{\text{inter}}$ |
| > 1 | Stronger "gravity of the species" | $\propto \text{MVN}_{\text{inter}}^{\,L}$ (narrower) |

$L$ is therefore a **tempering parameter on the species-level constraint**, not a fitted quantity.
It is held at 1 throughout, and is varied only as a diagnostic: with heating switched off, the
standard deviation of $\ln\delta$ after 100 cycles is 1.013 at $L = 0$, 0.233 at $L = 1$ (against
a target of 0.258) and 0.166 at $L = 2$, confirming the behavior tabulated above.

---

## 5. Simulation settings

| Symbol | Definition | Value used |
|---|---|---|
| $N_{\text{cell}}$ | Population size after regrowth | $10^4$ (4-log regrowth) |
| $n$ | **Carry-over parameter**: cells transferred between successive cycles | 1 |
| $t_{\text{heating}}$ | Heat treatment duration per cycle | 2.591 / 5.182 / 7.774 min |
| $t_{\text{death}}$ | Death time of an individual cell | sampled per cell |
| — | Number of heating–regrowth cycles | 100 |
| — | Number of independent replicate chains | 10,000 |
| — | Treatment temperature | 55 °C |

Individual death times are generated by inverse-transform sampling from the same modified Weibull
model, with $u \sim \text{Uniform}(0,1)$:

$$t_{\text{death}} \;=\; 6\delta\left(\frac{-\log_{10}(1-u)}{6}\right)^{1/p}$$

A cell survives the cycle if $t_{\text{death}} > t_{\text{heating}}$.

**On $n = 1$.** This represents the extreme case in which regrowth between cycles is initiated by a
single cell carried over in a droplet or by surface contact. It is the *conservative* setting: the
sensitivity analysis shows that larger $n$ produces *larger* shifts toward thermotolerance
(saturating for $n \gtrsim 100$), because selection can then act across lineages rather than being
dominated by drift within one. Conversely, $n = 1$ maximizes the probability of complete
population extinction.

---

## 6. Sensitivity analysis parameters

These multiply the **standard deviations**, so the corresponding covariance matrices are scaled by
the square.

| Symbol | Applies to | Values tested | Published setting |
|---|---|---|---|
| $s_{\text{intra}}$ | SD of $\text{MVN}_{\text{intra}}$ | 0.25, 0.5, 1, 2, 4 | 1 |
| $s_{\text{inter}}$ | SD of $\text{MVN}_{\text{inter}}$ | 0.5, 0.75, 1, 1.5, 2 | 1 |
| $n$ | Carry-over parameter | 1, 10, 100, 1000 | 1 |
| $L$ | Acceptance-step exponent | 0, 0.5, 1, 2 | 1 |

---

## 7. Manuscript notation and code variable names

| Manuscript | Code variable | Where |
|---|---|---|
| $\left(\ln\delta_{\text{mean}},\ \ln p_{\text{mean}}\right)$ | `samples['Paras0_between']` | Stan posterior |
| $\text{COV}_{\text{inter}}$ | `samples['cov_between']` | Stan posterior |
| $\text{COV}_{\text{intra}}$ | `samples['cov_within']` | Stan posterior |
| $L$ | `L_para` | simulation |
| $s_{\text{intra}}$ | `sigma_adjust_para` (original) / `sd_scale_within` (revision) | simulation |
| $s_{\text{inter}}$ | `sd_scale_between` (revision) | simulation |
| $N_{\text{cell}}$ | `cell_count` | simulation |
| $n$ | `n_carry` (revision; fixed at 1 in the original) | simulation |
| $t_{\text{heating}}$ | `t_heating` | simulation |
| Number of cycles | `revo_num` | simulation |
| Number of replicate chains | `sim_num` | simulation |

> **On `between` / `within` versus `inter` / `intra`.** The two pairs denote the same two levels of
> variability. The manuscript uses **inter** and **intra** throughout; the code retains
> `between` and `within` as variable names, inherited from the Stan model. They are
> interchangeable: `cov_between` = $\text{COV}_{\text{inter}}$ and
> `cov_within` = $\text{COV}_{\text{intra}}$.

---

## 8. Files

| File | Content |
|---|---|
| `Multiple_heating_and_revolution.ipynb` | Main simulation reported in the paper |
| `Revision_analyses.ipynb` | Algorithm validation and sensitivity analyses |
| `revision_core.py` | Generalized simulation engine (reduces exactly to the published chain at $n = 1$, $L = 1$, $s = 1$) |
| `Samples extracted modfied Weibull multi-level MVN stan.pkl` | Extracted Stan posterior draws |
| `modified Weibull multi-level MVN.pkl` | Fitted Stan model object |
