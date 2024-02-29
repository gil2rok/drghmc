from collections import defaultdict
import os

import numpy as np
import wandb

from samplers.drghmc import DrGhmcDiag
from utils.argument_parsers import drhmc_argument_parser
from utils.posteriors_new import get_model_path, get_data_path, get_init
from utils.models import BayesKitModel, grad_counter
from utils.nuts_utils import get_nuts_step_size, get_nuts_step_counts
from utils.metrics import compute_metrics
from utils.save_utils import save_to_npz, get_data_dir, save_fingerprint


def configure_sampler(config):    
    model_path = get_model_path(config.posterior, config.posterior_dir)
    data_path = get_data_path(config.posterior, config.posterior_dir)
    
    model = BayesKitModel(model_path=model_path, data_path=data_path)
    model.log_density_gradient = grad_counter(model.log_density_gradient)

    if config.step_size_factor:
        nuts_step_size = get_nuts_step_size(config)
        init_step_size = nuts_step_size * config.step_size_factor
    elif config.step_size:
        init_step_size = config.step_size

    step_sizes = [
        init_step_size * (config.reduction_factor**-k)
        for k in range(config.max_proposals)
    ]

    if config.step_count_factor:
        nuts_step_counts = get_nuts_step_counts(config)
        init_step_count = int(np.percentile(nuts_step_counts, config.step_count_factor * 100))
    elif config.step_count:
        init_step_count = config.step_count
        
    traj_len = init_step_count * init_step_size
    step_counts = [
        int(traj_len / step_sizes[k]) for k in range(config.max_proposals)
    ]

    if config.metric == 0:
        metric = get_nuts_metric(wandb.run.entity, wandb.run.project, f"nuts__chain-{config.chain}")
    elif config.metric == 1:
        metric = None

    # init = get_init(ref_draws, config.chain, "bk", posterior_origin)
    init = get_init(config.posterior, config.posterior_dir, config.chain)
    
    damping = float(config.damping) # w&b casts damping float 1.0 to 1

    return DrGhmcDiag(
        model=model,
        max_proposals=config.max_proposals,
        leapfrog_step_sizes=step_sizes,
        leapfrog_step_counts=step_counts,
        damping=damping,
        metric_diag=metric,
        init=init,
        seed=config.seed,
        prob_retry=config.probabilistic,
    )


def _compute_accept_prob(diagnostic):
    """From individual proposal acceptance probabilities, compute overall acceptance
    probability"""
    # extract acceptance probabilities from diagnostic dict
    acceptance_probs = []
    for k, v in diagnostic.items():
        if "acceptance" in k:
            acceptance_probs.append(v)

    # compute overall acceptance probability
    acceptance = 0
    for i, a in enumerate(acceptance_probs):
        term = a
        for j in range(i):
            term *= 1 - acceptance_probs[j]
        acceptance += term
    return acceptance


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


@run_once
def init_wandb(num_params):
    wandb.define_metric("grad_evals", summary="max")
    for i in range(num_params):
        wandb.define_metric(f"param_{i}/trace")


def log_history_to_wandb(history, idx=-1):
    cur_draw = history["draws"][idx]
    num_params = len(cur_draw)
    init_wandb(num_params)
    
    log_dict = {f"param_{i}/trace": param for i, param in enumerate(cur_draw)}
    log_dict["grad_evals"] = history["grad_evals"][idx]
    log_dict["acceptance_prob"] = history["accept_prob"][idx]
    wandb.log(log_dict)


def run_sampler(sampler, gradient_budget):
    history = defaultdict(list)
    log_freq, prev_log = max(1, gradient_budget // 10000), 0

    while sampler._model.log_density_gradient.calls < gradient_budget:
        # sample from the sampler
        draw = sampler.sample()[0]
        accept_prob = _compute_accept_prob(sampler.diagnostics)
        step_size = sampler._leapfrog_step_sizes[sampler.diagnostics["acceptance"] - 1]
        step_count = sampler._leapfrog_step_counts[
            sampler.diagnostics["acceptance"] - 1
        ]

        # record history
        history["draws"] += [draw]
        history["grad_evals"] += [sampler._model.log_density_gradient.calls]
        history["accept_prob"] += [accept_prob]
        history["acceptance"] += [sampler.diagnostics["acceptance"]]
        history["num_nans"] += [sampler.diagnostics["num_nans"]]
        history["step_size"] += [step_size]
        history["step_count"] += [step_count]
        
        # wandb logging (ensure log first and last draw)
        if sampler._model.log_density_gradient.calls - prev_log >= log_freq or prev_log == 0 or sampler._model.log_density_gradient.calls >= gradient_budget:
            log_history_to_wandb(history)
            prev_log = sampler._model.log_density_gradient.calls

    return history


def main(config):
    if config.generate_history:
        sampler = configure_sampler(config)
        history = run_sampler(sampler, config.gradient_budget)
        save_to_npz(dictionary=history, config=config, filename="history")
        save_fingerprint(config)

    if config.generate_metrics:
        path = os.path.join(get_data_dir(config), "history.npz")
        try:
            history = np.load(path)
        except Exception as e:
            raise FileNotFoundError(
                f"History file not found at\n\t{path}\nwith error:\n\t{e}"
            )

        metrics = compute_metrics(history, config)
        save_to_npz(dictionary=metrics, config=config, filename="metrics")


if __name__ == "__main__":
    parser = drhmc_argument_parser()
    args = parser.parse_args()

    irrelevant_hyperparams = [
        "experiment",
        "posterior",
        "posterior_dir",
        "seed",
        "sampler_type",
        "generate_history",
        "generate_metrics",
        "burn_in",
        "gradient_budget",
        "metric",
        "probabilistic",
    ]

    group_name = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and k not in ["chain"] and v is not None
        ]
    )
    run_name = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and v is not None
        ]
    )

    WANDB_RUN_GROUP = group_name  # set environment var
    wandb.init(
        config=args,
        # name=run_name,
        group=group_name,
        project=args.experiment,
        job_type=args.sampler_type,
        save_code=False,
    )
    main(wandb.config)
