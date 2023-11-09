import argparse
from collections import namedtuple

from tqdm import tqdm
from mpi4py import MPI
import numpy as np
from sklearn.model_selection import ParameterGrid

from src.utils.save_samples import my_save, stan_save, call_counter
from src.utils.configure_samplers import bayes_kit_hmc, bayes_kit_mala, stan_nuts, hmc, ghmc, drhmc, drghmc

HyperParamsTuple = namedtuple(
    "hyper_params",
    [
        "model_num",
        "chain_num",
        "burn_in_gradeval",
        "chain_length_gradeval",
        "global_seed",
        "save_dir",
        "pdb_dir",
        "bridgestan_dir",
    ],
)
SamplerParamsTuple = namedtuple(
    "model_params",
    [
        "init_stepsize",
        "reduction_factor",
        "steps",
        "dampening",
        "num_proposals",
        "probabilistic",
    ],
    defaults=[None, 2, 1, 0, 1, False],
)


def experiment(sampler, hp, burn_in, chain_len):
    # TODO: add multi-processing here
    burned_draws = np.asanyarray([sampler.sample()[0] for _ in range(burn_in)])
    
    sampler._model.log_density_gradient = call_counter(
        sampler._model.log_density_gradient
    )
    sampler._model.log_density = call_counter(sampler._model.log_density)
    
    # TODO: add multi-processing here
    draws = np.asanyarray([sampler.sample()[0] for _ in range(chain_len)])
    
    return burned_draws, draws


def stan_nuts_runner(hp):
    sampler_type = "nuts"
    
    model, data, seed, init, inv_metric = stan_nuts(hp)
    nuts_fit = model.sample(
        data=data,
        chains=1,
        seed=seed,
        inits=init,
        metric=inv_metric,
        #  adapt_init_phase=0  # b/c init from reference draw
    )
    
    stan_save(nuts_fit, sampler_type, hp)


def bayes_kit_hmc_runner(hp):
    sampler_type = "bk_hmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
            "steps": [20],
        }
    )

    for sampler_params in tqdm(sampler_param_grid, desc=sampler_type):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = bayes_kit_hmc(hp, sp)

        burn_in = int(hp.burn_in_gradeval / (sp.steps))
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)
        
        
def bayes_kit_mala_runner(hp):
    sampler_type = "bk_mala"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
        }
    )

    for sampler_params in tqdm(sampler_param_grid, desc=sampler_type):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = bayes_kit_mala(hp, sp)

        burn_in = int(hp.burn_in_gradeval / (sp.steps))
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def hmc_runner(hp):
    sampler_type = "hmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2, 5e-2, 1e-1, 2e-1],
            "steps": [10, 20, 30 ,50],
        }
    )

    for sampler_params in tqdm(sampler_param_grid, desc=sampler_type):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = hmc(hp, sp)
        
        burn_in = int(hp.burn_in_gradeval / (sp.steps))
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def ghmc_runner(hp):
    sampler_type = "ghmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2, 5e-2, 1e-1, 2e-1],
            "dampening": [0.01, 0.05, 0.1, 0.2],
        }
    )

    for sampler_params in tqdm(sampler_param_grid, desc=sampler_type):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = ghmc(hp, sp)

        burn_in = int(hp.burn_in_gradeval / sp.steps)
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def drhmc_runner(hp):
    sampler_type = "drhmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [0.12063],
            "reduction_factor": [2, 4],
            "steps": [70, 35],
            "num_proposals": [2, 3, 4],
            "probabilistic": [False],
        }
    )

    for sampler_params in tqdm(sampler_param_grid, desc=sampler_type):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = drhmc(hp, sp)

        burn_in = int(hp.burn_in_gradeval / sp.steps)
        #  chain_len = int(hp.chain_length_gradeval / sp.steps)
        chain_len = 1000

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def drghmc_runner(hp):
    sampler_type = "drghmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [0.12063],
            "reduction_factor": [2, 4],
            "steps": ["const_traj_len", 1],
            "dampening": [0.01, 0.05, 0.1, 0.2, 0.3],
            "num_proposals": [1, 2, 3, 4],
            "probabilistic": [False],
        }
    )

    for sampler_params in tqdm(sampler_param_grid, desc=sampler_type):
        sp = SamplerParamsTuple(**sampler_params)
        sampler = drghmc(hp, sp)

        burn_in = (
            int(hp.burn_in_gradeval / sp.steps)
            if type(sp.steps) is int
            else hp.burn_in_gradeval
        )
        # chain_len = (
        #     int(hp.chain_length_gradeval / sp.steps)
        #     if type(sp.steps) is int
        #     else hp.chain_length_gradeval
        # )
        chain_len = 1000

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", type=str, help="PDB model number")
    args = parser.parse_args()

    hp = HyperParamsTuple(
        model_num=args.model_num,
        chain_num=0,
        burn_in_gradeval=0,  # initialize with reference sample, don't require real burn-in
        chain_length_gradeval=500000,
        global_seed=MPI.COMM_WORLD.Get_rank(),  # represents an individual "run"
        save_dir="data/raw",
        pdb_dir="posteriors/",
        bridgestan_dir="../../.bridgestan/bridgestan-2.1.1/",
    )
    
    print(hp.global_seed)

    stan_nuts_runner(hp)
    #  hmc_runner(hp)
    #  ghmc_runner(hp)
    drhmc_runner(hp)
    drghmc_runner(hp)
