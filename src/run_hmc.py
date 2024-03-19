from collections import defaultdict
import logging
import os

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import wandb

from .samplers.drghmc import DrGhmcDiag
from .utils.delayed_rejection import compute_accept_prob
from .utils.logging import logging_context, log_history_to_wandb
from .utils.metrics import compute_metrics
from .utils.models import BayesKitModel, grad_counter
from .utils.nuts_utils import get_nuts_step_size, get_nuts_metric, get_nuts_step_counts
from .utils.posteriors_new import get_model_path, get_data_path, get_init
from .utils.save_utils import save_dict_to_npz, get_history

os.environ["WANDB__SERVICE_WAIT"] = "300"


def configure_sampler(sampler, posterior):
    model_path = get_model_path(posterior.name, posterior.dir)
    data_path = get_data_path(posterior.name, posterior.dir)

    model = BayesKitModel(model_path=model_path, data_path=data_path)
    model.log_density_gradient = grad_counter(model.log_density_gradient)

    if sampler.params.step_size_factor:
        nuts_step_size = get_nuts_step_size(sampler, posterior)
        step_size = [nuts_step_size * sampler.params.step_size_factor]
    elif sampler.params.step_size:
        step_size = [sampler.params.step_size]

    if sampler.params.step_count_factor:
        nuts_step_counts = get_nuts_step_counts(sampler, posterior)
        step_count = [int(
            np.percentile(nuts_step_counts, sampler.params.step_count_factor * 100)
        )]
    elif sampler.params.step_count:
        step_count = [sampler.params.step_count]
    
    logging.info(f"Step size: {step_size}")
    logging.info(f"Step count: {step_count}")

    if sampler.params.metric == 0:
        metric = get_nuts_metric(
            wandb.run.entity, wandb.run.project, f"nuts__chain-{sampler.chain}"
        )
    elif sampler.params.metric == 1:
        metric = None

    init = get_init(posterior.name, posterior.dir, sampler.chain)

    damping = float(sampler.params.damping)  # w&b casts damping float 1.0 to 1

    seed = int(str(sampler.seed) + str(sampler.chain))

    return DrGhmcDiag(
        model=model,
        max_proposals=1,
        leapfrog_step_sizes=step_size,
        leapfrog_step_counts=step_count,
        damping=damping,
        metric_diag=metric,
        init=init,
        seed=seed,
        prob_retry=sampler.params.probabilistic,
    )


def run_sampler(sampler, config):
    history = defaultdict(list)
    thin_counter = -1 # start at -1 to ensure first draw is not thinned
    gradient_budget = config.sampler.gradient_budget
    points_per_metric = config.wandb.points_per_metric
    log_freq, prev_log = max(1, gradient_budget // points_per_metric), 0

    while sampler._model.log_density_gradient.calls < gradient_budget:
        # sample from the sampler
        draw = sampler.sample()[0]
        accept_prob = compute_accept_prob(sampler.diagnostics)
        step_size = sampler._leapfrog_step_sizes[sampler.diagnostics["acceptance"] - 1]
        step_count = sampler._leapfrog_step_counts[
            sampler.diagnostics["acceptance"] - 1
        ]
        
        # thin draws
        thin_counter += 1
        if thin_counter % config.sampler.thin != 0:
            continue

        # record history
        history["draws"] += [draw]
        history["grad_evals"] += [sampler._model.log_density_gradient.calls]
        history["accept_prob"] += [accept_prob]
        history["acceptance"] += [sampler.diagnostics["acceptance"]]
        history["num_nans"] += [sampler.diagnostics["num_nans"]]
        history["step_size"] += [step_size]
        history["step_count"] += [step_count]

        # wandb logging (ensure log first and last draw)
        log_history_bool = (config.logging.log_metrics and config.logging.logger == "wandb")
        log_iter_bool = (sampler._model.log_density_gradient.calls - prev_log >= log_freq) or (prev_log == 0) or (sampler._model.log_density_gradient.calls >= gradient_budget)
        if log_history_bool and log_iter_bool:
            log_history_to_wandb(history)
            prev_log = sampler._model.log_density_gradient.calls

    return history


@hydra.main(version_base=None, config_path="../configs/samplers", config_name="hmc")
def main(config):
        
    if config.sampler.generate_history:
        with logging_context(config, job_type="history"):
            
            sampler = configure_sampler(config.sampler, config.posterior)
            history = run_sampler(sampler, config)
            save_dict_to_npz(dictionary=history, filename=f"history__chain={config.sampler.chain}")

    if config.sampler.generate_metrics:
        with logging_context(config, job_type="metrics"):

            history = get_history(config)
            metrics = compute_metrics(history, config)
            save_dict_to_npz(dictionary=metrics, filename=f"metrics__chain={config.sampler.chain}")
    
    ret = (
        metrics["max_se1"][-1], 
        metrics["max_se2"][-1],
        metrics["max_rse1"][-1], 
        metrics["max_rse2"][-1], 
    )
    return ret

if __name__ == "__main__":
    main()
