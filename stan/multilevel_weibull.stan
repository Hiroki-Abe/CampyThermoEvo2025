// Multilevel modified-Weibull model for thermal inactivation of C. jejuni
// (Equations 1a-1d of the manuscript), written for CmdStan >= 2.33.
//
//   (1a) N_plate ~ Poisson( 10^(log10 N_t) * V / 10^d )
//   (1b) log10 N_t = log10 N_0 - 6 * ( t / (6*delta) )^p
//   (1c) (ln delta, ln p)_curve  ~ MVN_intra( (ln delta, ln p)_strain , COV_intra )
//   (1d) (ln delta, ln p)_strain ~ MVN_inter( (ln delta, ln p)_mean   , COV_inter )
//
// One "curve" is one replicate experiment of one strain. Indexing by curve,
// rather than by a rectangular [strain, replicate] array, means that every
// replicate-level parameter is attached to its own plate counts and no
// parameter is left without data when strains have different numbers of
// replicates.
//
// Both levels use the centred form, as in the original PyStan 2.19 code. The
// plate counts pin down each curve very tightly (thousands of colonies per
// curve), and in that situation the centred form samples far better than the
// non-centred one. Unlike the original, (ln delta, ln p) are sampled directly
// on the log scale, so there are no upper bounds on delta (5) or p (2.5).
//
// Priors follow the original code: uniform on the species means within the
// same bounds, half-Cauchy(0, 1) on the standard deviations, LKJ(1) on the
// correlations, uniform on log10 N_0 within [logC0_lower, logC0_upper].

data {
  int<lower=1> N;                                   // number of plate counts
  int<lower=1> S;                                   // number of strains
  int<lower=1> J;                                   // number of curves (strain x replicate)
  array[J] int<lower=1, upper=S> strain_of_curve;   // strain index of each curve
  array[N] int<lower=1, upper=J> curve;             // curve index of each plate count
  vector<lower=0>[N] t_heat;                        // heating time (min)
  vector[N] n_dilu;                                 // number of 10-fold dilutions, d
  vector<lower=0>[N] V;                             // plated volume
  array[N] int<lower=0> count;                      // colony count
  real logC0_lower;                                 // bounds for log10 N_0 (original code: 6)
  real<lower=logC0_lower> logC0_upper;              //                       (original code: 10)
}

transformed data {
  real LOG6 = log(6.0);
  real LN10 = log(10.0);
  // Observations at t = 0 carry no reduction term; handling them separately
  // avoids the undefined derivative of 0^p with respect to p.
  int n_pos = 0;
  for (n in 1:N) n_pos += (t_heat[n] > 0);
  array[n_pos] int pos;
  {
    int i = 1;
    for (n in 1:N) if (t_heat[n] > 0) { pos[i] = n; i += 1; }
  }
  vector[n_pos] log_t_pos = log(t_heat[pos]);
  vector[N] log_V = log(V);
}

parameters {
  vector<lower=logC0_lower, upper=logC0_upper>[J] logC0;   // log10 N_0 of each curve

  real<lower=-5, upper=1.6> logdelta_mean;                 // ln delta_mean
  real<lower=-1, upper=1>   logpower_mean;                 // ln p_mean

  vector<lower=0>[2] sigma_inter;
  cholesky_factor_corr[2] Lcorr_inter;
  vector<lower=0>[2] sigma_intra;
  cholesky_factor_corr[2] Lcorr_intra;

  array[S] vector[2] theta_strain;   // (ln delta, ln p) strain means, Eq. 1d
  array[J] vector[2] theta_curve;    // (ln delta, ln p) of each replicate curve, Eq. 1c
}

transformed parameters {
  vector[2] mu_inter = [logdelta_mean, logpower_mean]';
}

model {
  // ---- likelihood (Eq. 1a, 1b) ----------------------------------------
  {
    vector[N] log10_N = logC0[curve] - n_dilu;
    vector[J] lnd;
    vector[J] lnp;
    for (j in 1:J) { lnd[j] = theta_curve[j, 1]; lnp[j] = theta_curve[j, 2]; }
    // 6 * (t / (6 delta))^p  computed on the log scale for t > 0
    log10_N[pos] -= 6 * exp( exp(lnp[curve[pos]])
                             .* (log_t_pos - LOG6 - lnd[curve[pos]]) );
    count ~ poisson_log(LN10 * log10_N + log_V);
  }

  // ---- hierarchical structure (Eq. 1c, 1d) ----------------------------
  theta_strain ~ multi_normal_cholesky(mu_inter, diag_pre_multiply(sigma_inter, Lcorr_inter));
  theta_curve  ~ multi_normal_cholesky(theta_strain[strain_of_curve],
                                       diag_pre_multiply(sigma_intra, Lcorr_intra));

  // ---- priors ----------------------------------------------------------
  sigma_inter ~ cauchy(0, 1);
  sigma_intra ~ cauchy(0, 1);
  Lcorr_inter ~ lkj_corr_cholesky(1);
  Lcorr_intra ~ lkj_corr_cholesky(1);
}

generated quantities {
  // Names chosen to match the keys used by the simulation code
  // (Paras0_between, cov_between, cov_within in the original pickle).
  matrix[2, 2] cov_inter  = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma_inter, Lcorr_inter));
  matrix[2, 2] cov_intra  = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma_intra, Lcorr_intra));
  real corr_inter = cov_inter[1, 2] / sqrt(cov_inter[1, 1] * cov_inter[2, 2]);
  real corr_intra = cov_intra[1, 2] / sqrt(cov_intra[1, 1] * cov_intra[2, 2]);
}
