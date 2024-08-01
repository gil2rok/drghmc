data {
    int<lower=1> N;          // Dimension of the multivariate normal distribution
    int<lower=1> M;          // Number of data points
    matrix[N, M] y;          // Observed data points
}

parameters {
    real<lower=-1, upper=1>[N] rho;
}

transformed parameters {
    L = 
}

model {
    // Prior
    L ~ lkj_corr_cholesky(4);
    

    // Likelihood
    y ~ multi_normal_cholesky(rep_vector(0, N), L);
}
