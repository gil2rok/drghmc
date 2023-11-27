import argparse
from collections import namedtuple

from tqdm import tqdm
from mpi4py import MPI
import numpy as np
from sklearn.model_selection import ParameterGrid

from src.utils.configure_samplers import stan_nuts, ghmc, drhmc, drghmc
from src.utils.save_samples import bayeskit_save, stan_save, grad_counter
from src.utils.summary_stats import get_summary_stats

HyperParamsTuple = namedtuple(
    "hyper_params",
    [
        "posterior",
        "burn_in",
        "chain_len",
        "global_seed",
        "chain",
        "save_dir",
        "posterior_dir",
        "bridgestan_dir",
    ],
)


SamplerParamsTuple = namedtuple(
    "sampler_params",
    [
        "sampler_type",
        "init_stepsize",
        "reduction_factor",
        "steps",
        "dampening",
        "num_proposals",
        "probabilistic",
    ],
    defaults=[None, None, 2, 1, 0, 1, False],
)


def generate_draws(sampler, hp):
    # TODO: add multi-processing here
    burned_draws = np.asanyarray([sampler.sample()[0] for _ in range(hp.burn_in)])
    
    sampler._model.log_density_gradient = grad_counter(
        sampler._model.log_density_gradient
    )
    
    # TODO: add multi-processing here
    draws = np.asanyarray([sampler.sample()[0] for _ in range(hp.chain_len)])
    
    return burned_draws, draws


def stan_nuts_runner(hp):
    model, data, seed, init, inv_metric = stan_nuts(hp)
    nuts_fit = model.sample(
        data=data,
        chains=1,
        seed=seed,
        inits=init,
        metric=inv_metric,
        #  adapt_init_phase=hp.burn_in  # b/c init from reference draw
    )
    
    draws = nuts_fit.draws(concat_chains=True)[:, 7:]
    summary_stats = get_summary_stats(draws)  # [draws, params]
    stan_save(nuts_fit, hp, summary_stats)


def ghmc_runner(hp):
    sampler_param_grid = ParameterGrid(
        {
            "sampler_type": ["ghmc"],
            "init_stepsize": [1e-2, 5e-2, 1e-1, 2e-1],
            "dampening": [0.01, 0.05, 0.1, 0.2],
        }
    )

    for idx, sampler_params in enumerate(tqdm(sampler_param_grid, desc="ghmc")):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = ghmc(hp, sp)
        burned_draws, draws = generate_draws(sampler, hp)
        summary_stats = get_summary_stats(draws)
        bayeskit_save(sp, hp, draws, sampler, idx, summary_stats)


def drhmc_runner(hp):
    sampler_param_grid = ParameterGrid(
        {
            "sampler_type": ["drhmc"],
            "init_stepsize": [0.12063],
            "reduction_factor": [2, 4],
            "steps": [70, 35],
            "num_proposals": [2, 3, 4],
            "probabilistic": [False],
        }
    )

    for idx, sampler_params in enumerate(tqdm(sampler_param_grid, desc="drhmc")):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = drhmc(hp, sp)
        burned_draws, draws = generate_draws(sampler, hp)
        summary_stats = get_summary_stats(draws)
        bayeskit_save(sp, hp, draws, sampler, idx, summary_stats)


def drghmc_runner(hp):
    sampler_param_grid = ParameterGrid(
        {
            "sampler_type": ["drghmc"],
            "init_stepsize": [0.12063],
            "reduction_factor": [2, 4],
            "steps": ["const_traj_len", 1],
            "dampening": [0.01, 0.05, 0.1, 0.2, 0.3],
            "num_proposals": [2, 3, 4],
            "probabilistic": [False],
        }
    )

    for idx, sampler_params in enumerate(tqdm(sampler_param_grid, desc="drghmc")):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = drghmc(hp, sp)
        burned_draws, draws = generate_draws(sampler, hp)
        summary_stats = get_summary_stats(draws)
        bayeskit_save(sp, hp, draws, sampler, idx, summary_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posterior", 
        type=str, 
        help="name of a posterior, specified by a Stan model and data"
    )
    args = parser.parse_args()

    hp = HyperParamsTuple(
        posterior=args.posterior,
        burn_in=0,  # initialize with reference sample, don't require real burn-in
        chain_len=1000,
        global_seed=0,
        chain=MPI.COMM_WORLD.Get_rank(),  # represents an individual "run"
        save_dir="data/raw",
        posterior_dir="posteriors/",
        bridgestan_dir="../../.bridgestan/bridgestan-2.1.1/",
    )
    
    stan_nuts_runner(hp)
    # ghmc_runner(hp)
    # drhmc_runner(hp)
    # drghmc_runner(hp)
