transformed data {
    vector[50] mu1 = append_row(0, rep_vector(0, 49));
    vector[50] mu2 = append_row(8, rep_vector(0, 49));
}
parameters {
    vector[50] y;
}
model {
    target += log_mix(0.3,
    normal_lpdf(y | 0, 1),
    normal_lpdf(y | 8, 1));
}