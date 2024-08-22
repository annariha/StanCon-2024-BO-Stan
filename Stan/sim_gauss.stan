data {
  int<lower=1> N;
  array[N] real x;
  real<lower=0> alpha;
  real<lower=0> rho;
  real mu;
  real<lower=0> sigma;
}

parameters {
  vector[N] eta;
}

transformed parameters{
  matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho) + 
                   diag_matrix(rep_vector(1e-10, N));
  matrix[N, N] L = cholesky_decompose(K);
}

model {
  eta ~ std_normal();
}

generated quantities {
  vector[N] y;
  vector[N] g = mu + L * eta;
  for (n in 1:N)
    y[n] = normal_rng(g[n], sigma);
}
