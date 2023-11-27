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


def stan_nuts(hp):
    model, data, ref_draws = get_posterior(hp.posterior, hp.posterior_dir, "stan")

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain].items():
        init[param_name] = param_value[-1]
        metric.append(np.std(param_value))
    inv_metric = {"inv_metric": [1 / el for el in metric]}

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain))
    
    return model, data, seed, init, inv_metric
        

def ghmc(hp, sp):
    model, ref_draws = get_posterior(hp.posterior, hp.posterior_dir, "bayeskit")
    stan_metric, stan_stepsize, stan_steps = get_stan_params(hp)

    init_stepsize = stan_stepsize * sp.init_stepsize
    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain))

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain].items():
        init[param_name] = param_value[-1]
        metric.append(np.std(param_value))
    init = np.array(list(init.values()), dtype=np.float64)

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
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain))

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain].items():
        init[param_name] = param_value[-1]
        metric.append(np.std(param_value))
    init = np.array(list(init.values()), dtype=np.float64)
    metric = stan_metric

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

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain))

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain].items():
        init[param_name] = param_value[-1]
        metric.append(np.std(param_value))
    init = np.array(list(init.values()), dtype=np.float64)
    metric = stan_metric

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
