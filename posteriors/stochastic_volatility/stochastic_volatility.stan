data {
    int<lower=0> T;   // # time points (equally spaced)
    vector[T] y;      // mean corrected return at time t
}
parameters {
    real mu;                     // mean log volatility
    real<lower=-1,upper=1> phi;  // persistence of volatility
    real<lower=0> sigma;         // white noise shock scale
    vector<lower=0>[T] tau;      // log volatility at time t
}
model {
    phi ~ uniform(-1,1);
    sigma ~ cauchy(0,2.5);
    mu ~ cauchy(0,10);
    tau[1] ~ lognormal(mu, sigma / sqrt(1 - phi * phi));
    tau[2:T] ~ lognormal(mu + phi * (tau[1:T - 1] - mu), sigma);
    y ~ normal(0, tau);
}
