data {
    int<lower=0> D;
}
parameters {
    vector[D] x; 
    vector[D] y;
}

model {
x ~ normal(1, 1);
y ~ normal(x^2, 0.1);
}