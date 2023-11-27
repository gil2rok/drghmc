import json, os

import numpy as np
import pandas as pd

from ..drghmc import DrGhmcDiag
from .posteriors import get_posterior


def get_stan_params(hp):
    stan_path = os.path.join(hp.save_dir, hp.posterior, "nuts", f"chain_{hp.chain:02d}")
    stan_params = json.load(open(os.path.join(stan_path, "params.json"), "r"))
    stan_df = pd.read_csv(os.path.join(stan_path, "draws.csv"), sep="\t")
    
    metric = [1/x for x in stan_params["inv_metric"]]
    stepsize = stan_params["stepsize"]
    num_steps_p90 = np.percentile(stan_df["n_leapfrog__"], 90)
    
    return metric, stepsize, num_steps_p90


def get_metric(ref_draws, chain_num, sampler_type, posterior_origin):
    num_chains = len(ref_draws)
    if posterior_origin == "custom" and not chain_num < num_chains:
        raise ValueError(f"Invalid chain number {chain_num} for {posterior_origin} posterior")
    
    param_dict = ref_draws[chain_num % num_chains]
    inv_metric = list()
    
    for param_name, param_value in param_dict.items():
        inv_metric.append(1 / np.std(param_value))
        
    return inv_metric


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
        raise ValueError(f"Invalid method {method}")
      
    return init


def stan_nuts(hp):
    model, data, ref_draws, posterior_origin = get_posterior(hp.posterior, hp.posterior_dir, "stan")

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain))
    
    init = get_init(ref_draws, hp.chain, "stan", posterior_origin)

    inv_metric = get_metric(ref_draws, hp.chain, "stan", posterior_origin)

    return model, data, seed, init, inv_metric
        

def ghmc(hp, sp):
    model, ref_draws, posterior_origin = get_posterior(hp.posterior, hp.posterior_dir, "bayeskit")
    stan_metric, stan_stepsize, stan_steps = get_stan_params(hp)

    init_stepsize = stan_stepsize * sp.init_stepsize
    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]
    
    seed = int(str(hp.global_seed) + str(hp.chain))
    
    metric = stan_metric
    
    init = get_init(ref_draws, hp.chain, "bk", posterior_origin)

    return DrGhmcDiag(
        model=model,
        max_proposals=1,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=[1],
        damping=sp.dampening,
        metric_diag=metric,
        init=init,
        seed=seed,
        prob_retry=False,
    )


def drhmc(hp, sp):
    model, ref_draws = get_posterior(hp.posterior, hp.posterior_dir, "bayeskit")
    stan_metric, stan_stepsize, stan_steps = get_stan_params(hp)
    
    init_stepsize = stan_stepsize * sp.init_stepsize
    stepsize = [
        init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]

    init_steps = stan_steps if stan_steps else sp.steps
    traj_len = init_steps * stepsize[0]
    steps = [
        int(traj_len / stepsize[k]) for k in range(sp.num_proposals)
    ]  # const traj len
    
    seed = int(str(hp.global_seed) + str(hp.chain))
    
    metric = stan_metric
    
    init = get_init(ref_draws, hp.chain, "bk", posterior_origin)

    return DrGhmcDiag(
        model=model,
        max_proposals=sp.num_proposals,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=steps,
        damping=1.0,
        metric_diag=metric,
        init=init,
        seed=seed,
        prob_retry=sp.probabilistic,
    )


def drghmc(hp, sp):
    model, ref_draws = get_posterior(hp.posterior, hp.posterior_dir, "bayeskit")
    stan_metric, stan_stepsize, stan_steps = get_stan_params(hp)
    
    init_stepsize = stan_stepsize * sp.init_stepsize
    stepsize = [
        init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]

    if sp.steps == 1:
        # const number of steps (ghmc)
        steps = [sp.steps for _ in range(sp.num_proposals)]
    elif sp.steps == "const_traj_len":
        # const trajectory length (drhmc)
        init_steps = 1
        traj_len = init_steps * stepsize[0]
        steps = [int(traj_len / stepsize[k]) for k in range(sp.num_proposals)]
    else:
        raise ValueError("Invalid value for DRGHMC steps")

    seed = int(str(hp.global_seed) + str(hp.chain))

    metric = stan_metric
    
    init = get_init(ref_draws, hp.chain, "bk", posterior_origin)

    return DrGhmcDiag(
        model=model,
        max_proposals=sp.num_proposals,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=steps,
        damping=sp.dampening,
        metric_diag=metric,
        init=init,
        seed=seed,
        prob_retry=sp.probabilistic,
    )
