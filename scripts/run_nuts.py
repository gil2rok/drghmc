import logging
import multiprocessing
import os

import cmdstanpy as cmd
import hydra
import numpy as np
from omegaconf import OmegaConf

from src.utils.logging import (
    logging_context,
    log_history_to_wandb,
) 
from src.utils.metrics import compute_metrics
from src.utils.posteriors import (
    get_model_path,
    get_data_path,
    get_init,
    get_param_names,
    get_diagonal_covariance,
)
from src.utils.save_utils import save_dict_to_npz, get_history
from src.utils.summary_utils import write_summary


def configure_sampler(sampler, posterior):
    model_path = get_model_path(posterior.name, posterior.dir)
    model = cmd.CmdStanModel(
        stan_file=model_path, 
        cpp_options={"STAN_THREADS": True, "TBB_CXX_TYPE": "gcc"}
    )
    
    data_path = get_data_path(posterior.name, posterior.dir)

    seed = int(str(sampler.seed) + str(sampler.chain))

    inits = get_init(posterior.name, posterior.dir, sampler.chain)
    print(f"Init Shape: {inits.shape}")
    logging.info(f"Init Shape: {inits.shape}")
    param_names = get_param_names(posterior.name, posterior.dir)
    init = {param_name: init for param_name, init in zip(param_names, inits)}
    
    #### stochastic volatility
    # expected: [num_chains, num_params, num_draws]
    # actual: [num_chains, 1, num_params]
    
    #### rosenbrock
    # expected: [num_chains, num_params, num_draws]
    # actual:  [num_params]
    
    if sampler.params.metric == "diag_cov":
        diag_cov = get_diagonal_covariance(posterior.name, posterior.dir)
        inv_metric = {"inv_metric": diag_cov}
    elif sampler.params.metric == "identity":
        inv_metric = {"inv_metric": np.ones(len(param_names))}

    return model, data_path, seed, init, inv_metric


def run_sampler(config, model, data_path, seed, init, inv_metric, burn_in):
    logging.info(f"Running NUTS sampler")
    nuts = model.sample(
        data=data_path,
        chains=1,
        seed=seed,
        inits=init,
        thin=config.sampler.thin,
        iter_sampling=config.sampler.gradient_budget,
        metric=inv_metric,
        # do not need burn-in to converge towards typical set b/c init from ref draw
        adapt_init_phase=config.sampler.burn_in, 
        adapt_metric_window= None if config.sampler.params.adapt_metric else 0,
    )
    return nuts


def enforce_gradient_budget(nuts, gradient_budget):
    nuts_np = nuts.draws(concat_chains=True)
    grad_evals = np.cumsum(nuts_np[:, 4])
    try:
        idx = np.argwhere(grad_evals > gradient_budget)[0][0]
    except IndexError:
        logging.warning(
            f"Gradient budget {gradient_budget} not reached in {len(grad_evals)} gradient evaluations for NUTS"
        )
        idx  = -1
    return nuts_np[:idx, :]


def format_history(cmdstan_history, config):
    nuts_np = enforce_gradient_budget(cmdstan_history, config.sampler.gradient_budget)
    history = {
        "draws": nuts_np[:, 7:],
        "grad_evals": list(np.cumsum(nuts_np[:, 4], dtype=np.uint)),  # cumulative gradient evaluations
        "accept_prob": nuts_np[:, 1],
        "acceptance": [1]
        + [
            int(np.all(np.isclose(draw1, draw2)))
            for draw1, draw2 in zip(nuts_np[:, 7:-1], nuts_np[:, 8:])
        ],
        "num_nans": [0] * len(nuts_np[:, 4]),  # no NaNs in NUTS draws
        "step_size": nuts_np[:, 2],
        "step_count": nuts_np[:, 4],
        "metric": cmdstan_history.metric, # NUTS specific field used in DR samplers
    }

    # wandb logging
    gradient_budget = config.sampler.gradient_budget
    points_per_metric = config.wandb.points_per_metric
    log_freq, prev_log = max(1, gradient_budget // points_per_metric), 0
    total_grads = history["grad_evals"][-1]

    for idx in range(len(history["grad_evals"])):
        grad = history["grad_evals"][idx]

        log_history_bool = (
            config.logging.log_history and config.logging.logger == "wandb"
        )
        log_iter_bool = (
            (grad - prev_log >= log_freq) or (prev_log == 0) or (grad == total_grads)
        )
        if log_history_bool and log_iter_bool:
            log_history_to_wandb(history, idx)
            prev_log = grad

    return history

    
def worker(config, chain):
    os.sched_setaffinity(0, {chain % multiprocessing.cpu_count()})
    OmegaConf.update(config, "sampler.chain", chain)

    if config.sampler.generate_history:
        with logging_context(config, job_type="history"):

            model, data_path, seed, init, inv_metric = configure_sampler(
                config.sampler, config.posterior
            )
            cmdstan_history = run_sampler(
                config,
                model,
                data_path,
                seed,
                init,
                inv_metric,
                config.sampler.burn_in,
            )
            history = format_history(cmdstan_history, config)
            save_dict_to_npz(dictionary=history, filename=f"history__chain={config.sampler.chain}")

    if config.sampler.generate_metrics:
        with logging_context(config, job_type="metrics"):

            history = get_history(config)
            metrics = compute_metrics(history, config)
            save_dict_to_npz(dictionary=metrics, filename=f"metrics__chain={config.sampler.chain}")
            write_summary(config, metrics)


@hydra.main(version_base=None, config_path="../configs/samplers", config_name="nuts")
def main(config):
    processes = []
    
    for chain in range(config.sampler.chains):
        p = multiprocessing.Process(target=worker, args=(config, chain))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()