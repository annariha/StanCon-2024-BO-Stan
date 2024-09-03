
data {
  int<lower=1> N_obs;
  array[N_obs] real x_obs;
  vector[N_obs] y_obs;
  int<lower=1> N_pred;
  array[N_pred] real x_pred;
}

transformed data {
  int<lower=1> N = N_obs + N_pred;
  array[N] real x;
  for (n_obs in 1:N_obs)   x[n_obs] = x_obs[n_obs];
  for (n_pred in 1:N_pred) x[N_obs + n_pred] = x_pred[n_pred];
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  real mu;
  vector[N] eta;
}

transformed parameters{
  vector[N] g;
  { 
    matrix[N, N] L;
    matrix[N, N] K;
    K = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(1e-10, N));
    L = cholesky_decompose(K);
    g = mu + L * eta;
  }
}

model {
  rho   ~ normal(0.3,0.1);
  alpha ~ std_normal();
  sigma ~ std_normal();
  mu    ~ std_normal();
  eta   ~ std_normal();
  y_obs ~ normal(g[1:N_obs], sigma);
}

generated quantities {
  vector[N_pred] y_pred;
  for (n_pred in 1:N_pred){
    y_pred[n_pred] = normal_rng(g[N_obs + n_pred], sigma);
  }
}


