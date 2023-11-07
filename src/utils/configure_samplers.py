import os

import numpy as np

from ..drghmc import DrGhmcDiag
from .posteriors import get_posterior

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

    init = dict()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))
    
    return model, data, seed, init


def hmc(hp, sp):
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")

    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))
    
    init = dict()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]
    init = np.array(list(init.values()), dtype=np.float64)


    return DrGhmcDiag(
        model=model,
        max_proposals=1,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=[sp.steps],
        damping=1.0,
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

    init = dict()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]
    init = np.array(list(init.values()), dtype=np.float64)

    return DrGhmcDiag(
        model=model,
        max_proposals=1,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=[1],
        damping=sp.dampening,
        init=init,
        seed=seed,
        prob_retry=False,
    )


def drhmc(hp, sp):
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")
    
    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]

    traj_len = sp.steps * stepsize[0]
    steps = [
        int(traj_len / stepsize[k]) for k in range(sp.num_proposals)
    ]  # const traj len
    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    init = dict()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]
    init = np.array(list(init.values()), dtype=np.float64)

    return DrGhmcDiag(
        model=model,
        max_proposals=sp.num_proposals,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=steps,
        damping=1.0,
        init=init,
        seed=seed,
        prob_retry=sp.probabilistic,
    )


def drghmc(hp, sp):
    model, ref_draws = get_posterior(hp.model_num, hp.pdb_dir, "bayeskit")
    
    stepsize = [
        sp.init_stepsize * (sp.reduction_factor**-k) for k in range(sp.num_proposals)
    ]

    if sp.steps == 1:
        # const number of steps (ghmc)
        steps = [sp.steps for k in range(sp.num_proposals)]
    elif sp.steps == "const_traj_len":
        # const trajectory length (drhmc)
        init_steps = 1
        traj_len = init_steps * stepsize[0]
        steps = [int(traj_len / stepsize[k]) for k in range(sp.num_proposals)]
    else:
        raise ValueError("Invalid value for DRGHMC steps")

    # seed depends on global seed and chain number
    seed = int(str(hp.global_seed) + str(hp.chain_num))

    init = dict()
    for param_name, param_value in ref_draws[hp.chain_num].items():
        init[param_name] = param_value[-1]
    init = np.array(list(init.values()), dtype=np.float64)

    return DrGhmcDiag(
        model=model,
        max_proposals=sp.num_proposals,
        leapfrog_step_sizes=stepsize,
        leapfrog_step_counts=steps,
        damping=sp.dampening,
        init=init,
        seed=seed,
        prob_retry=sp.probabilistic,
    )
