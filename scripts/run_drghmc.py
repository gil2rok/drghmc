import multiprocessing
import os
from collections import defaultdict

import hydra
from omegaconf import OmegaConf

from src.samplers.drghmc import DrGhmcDiag
from src.utils.delayed_rejection import compute_accept_prob
from src.utils.logging import logging_context, log_history_to_wandb
from src.utils.metrics import compute_metrics
from src.utils.models import BayesKitModel, grad_counter
from src.utils.nuts_utils import get_nuts_step_size, get_nuts_metric
from src.utils.posteriors import get_model_path, get_data_path, get_init
from src.utils.save_utils import save_dict_to_npz, get_history
from src.utils.summary_utils import write_summary

os.environ["WANDB__SERVICE_WAIT"] = "300"


def configure_sampler(sampler, posterior):
    model_path = get_model_path(posterior.name, posterior.dir)
    data_path = get_data_path(posterior.name, posterior.dir)

    model = BayesKitModel(model_path=model_path, data_path=data_path)
    model.log_density_gradient = grad_counter(model.log_density_gradient)
    
    init = get_init(posterior.name, posterior.dir, sampler.chain)

    if sampler.params.step_size_factor:
        nuts_step_size = get_nuts_step_size(sampler, posterior)
        init_step_size = nuts_step_size * sampler.params.step_size_factor
    elif sampler.params.step_size:
        init_step_size = sampler.params.step_size

    step_sizes = [
        init_step_size * (sampler.params.reduction_factor**-k)
        for k in range(sampler.params.max_proposals)
    ]

    if sampler.params.step_count_method == "const_step_count":
        # const number of steps (ghmc)
        step_counts = [1 for _ in range(sampler.params.max_proposals)]
    elif sampler.params.step_count_method == "const_traj_length":
        # const trajectory length (drhmc)
        init_step_count = 1
        traj_len = init_step_count * init_step_size
        step_counts = [
            int(traj_len / step_sizes[k]) for k in range(sampler.params.max_proposals)
        ]

    metric = get_nuts_metric(sampler, posterior)

    damping = float(sampler.params.damping)  # w&b casts damping float 1.0 to 1

    seed = int(str(sampler.seed) + str(sampler.chain))

    return DrGhmcDiag(
        model=model,
        max_proposals=sampler.params.max_proposals,
        leapfrog_step_sizes=step_sizes,
        leapfrog_step_counts=step_counts,
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


def worker(config, chain):
    os.sched_setaffinity(0, {chain % multiprocessing.cpu_count()})
    OmegaConf.update(config, "sampler.chain", chain)
    
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
            write_summary(config, metrics)
            

@hydra.main(version_base=None, config_path="../configs/samplers", config_name="drghmc")
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
