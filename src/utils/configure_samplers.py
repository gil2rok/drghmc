import numpy as np

from utils.posteriors import get_posterior





def get_metric(ref_draws, chain_num, sampler_type, posterior_origin):
    num_chains = len(ref_draws)
    if posterior_origin == "custom" and not chain_num < num_chains:
        raise ValueError(f"Invalid chain number {chain_num} for {posterior_origin} posterior")
    
    param_dict = ref_draws[chain_num % num_chains]
    metric = list()
    
    for param_name, param_value in param_dict.items():
        metric.append(np.var(param_value))

    # inv_metric = {"inv_metric": [1/x for x in metric]}
    # return inv_metric
    metric = {"inv_metric": metric}
    return metric


def get_init(ref_draws, chain_num, sampler_type, posterior_origin):
    num_chains = len(ref_draws)
    if posterior_origin == "custom" and not chain_num < num_chains:
        raise ValueError(f"Invalid chain number {chain_num} for {posterior_origin} posterior")
    
    param_dict = ref_draws[chain_num % num_chains]
    init = dict()
    
    for param_name, param_value in param_dict.items():
        init[param_name] = param_value[-1 - (chain_num // num_chains) * 10]
        
    if sampler_type == "bk": # bayeskit expects numpy array for initialization, not dict
        init = np.array(list(init.values()), dtype=np.float64)
        return init
    elif sampler_type != "stan":
        raise ValueError(f"Invalid method {sampler_type}")

    return init


def stan_nuts(hp):
    model, data, ref_draws, posterior_origin = get_posterior(hp.posterior, hp.posterior_dir, "stan")

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain))
    
    init = get_init(ref_draws, hp.chain, "stan", posterior_origin)

    inv_metric = get_metric(ref_draws, hp.chain, "stan", posterior_origin)

    return model, data, seed, init, inv_metric