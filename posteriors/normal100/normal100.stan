data {
    int<lower=0> N;
    real<lower=0, upper=1> rho;
}
transformed data {
    vector[N] mu = rep_vector(0, N);
    matrix[N, N] Sigma;
    for (i in 1:N) for (j in 1:N) Sigma[i, j] = rho^abs(i - j);
    matrix[N, N] L_Sigma = cholesky_decompose(Sigma);
}
parameters {
    vector[N] y;
}
model {
    y ~ multi_normal_cholesky(mu, L_Sigma);
}