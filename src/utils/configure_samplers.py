import json, os

import numpy as np
import pandas as pd

from ..drghmc import DrGhmcDiag
from .posteriors import get_posterior


def get_stan_params(hp):
    path = os.path.join("data/raw/", hp.model_num)
    all_dirs = os.listdir(path)
    
    stan_path = None
    for dir in all_dirs:
        if dir.startswith("nuts"):
            stan_path = os.path.join(
                path, dir, 
                f"chain_{hp.chain_num:02d}", 
                f"run_{hp.global_seed:02d}"
            )
            break
    
    if not stan_path:
        raise ValueError("Bayes-Kit sampler must be called after Stan sampler")
    
    stan_params = json.load(open(os.path.join(stan_path, "sampler_params.json"), "r"))
    stan_df = pd.read_csv(os.path.join(stan_path, "draws.csv"), sep="\t")
    
    metric = [1/x for x in stan_params["metric"]]
    stepsize = stan_params["stepsize"]
    num_steps_p90 = np.percentile(stan_df["n_leapfrog__"], 90)
    
    return metric, stepsize, num_steps_p90


def bayes_kit_hmc(hp, sp):
    model = get_model(hp.model_num, hp.pdb_dir)
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    # remove after testing
    fname = os.path.join(
        hp.pdb_dir, f"PDB_{hp.model_num:02d}", f"PDB_{hp.model_num:02d}.samples.npy"
    )
    init_constrained = np.load(fname)[hp.chain_num - 1, -1, :].copy(order="C")
    init = model.unconstrain(init_constrained)

    return HMCDiag(
        model=model,
        stepsize=sp.init_stepsize,
        steps=sp.steps,
        init=None,
        seed=seed,
    )


def bayes_kit_mala(hp, sp):
    model = get_model(hp.model_num, hp.pdb_dir)
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    # remove after testing
    fname = os.path.join(
        hp.pdb_dir, f"PDB_{hp.model_num:02d}", f"PDB_{hp.model_num:02d}.samples.npy"
    )
    init_constrained = np.load(fname)[hp.chain_num - 1, -1, :].copy(order="C")
    init = model.unconstrain(init_constrained)

    return MALA(model=model, epsilon=sp.init_stepsize, init=None, seed=seed)


def stan_nuts(hp):
    model, data, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "stan")


    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]
        metric.append(np.std(param_value))
    inv_metric = {"inv_metric": [1 / el for el in metric]}

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))
    
    return model, data, seed, init, inv_metric


def hmc(hp, sp):
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")

    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))
    
    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]
        metric.append(np.std(param_value))
    init = np.array(list(init.values()), dtype=np.float64)


    return DrGhmcDiag(
        model=model,
        max_proposals=1,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=[sp.steps],
        damping=1.0,
        metric_diag=metric,
        init=init,
        seed=seed,
        prob_retry=False,
    )


def ghmc(hp, sp):
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")

    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain_num].items():
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
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")
    stan_metric, stan_stepsize, stan_steps = get_stan_params(hp)
    
    init_stepsize = stan_stepsize if stan_stepsize else sp.init_stepsize
    stepsize = [
        init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]

    init_steps = stan_steps if stan_steps else sp.steps
    traj_len = init_steps * stepsize[0]
    steps = [
        int(traj_len / stepsize[k]) for k in range(sp.num_proposals)
    ]  # const traj len
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain_num].items():
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
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")
    stan_metric, stan_stepsize, stan_steps = get_stan_params(hp)
    
    init_stepsize = stan_stepsize if stan_stepsize else sp.init_stepsize
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
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    init, metric = dict(), list()
    for param_name, param_value in ref_draws[hp.chain_num].items():
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
