data {
    int<lower=0> N;
    real<lower=0> nu;
    real alpha;
}
parameters {
    vector[N] beta;
}
model {
    beta ~ student_t(nu, 0, 1);
}
generated quantities {
    int<lower=0, upper=1> beta_lt_alpha = beta[1] <= alpha;
}