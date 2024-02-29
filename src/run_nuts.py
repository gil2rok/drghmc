from collections import namedtuple
import os

import cmdstanpy as cmd
import numpy as np
import polars as pl
import wandb

from utils.posteriors_new import get_model_path, get_data_path, get_init, get_param_names, get_diagonal_covariance
from utils.argument_parsers import nuts_argument_parser
from utils.metrics import compute_metrics
from utils.save_utils import get_data_dir, save_to_npz, save_fingerprint


def configure_sampler(config):    
    model_path = get_model_path(config.posterior, config.posterior_dir)
    model = cmd.CmdStanModel(stan_file=model_path)
    
    data_path = get_data_path(config.posterior, config.posterior_dir)
    
    seed = config.seed
    
    inits = get_init(config.posterior, config.posterior_dir, config.chain)
    param_names = get_param_names(config.posterior, config.posterior_dir)
    init = {param_name : init for param_name, init in zip(param_names, inits)}
    
    if config.metric == "diag_cov":
        diag_cov = get_diagonal_covariance(config.posterior, config.posterior_dir)
        inv_metric = {"inv_metric": diag_cov}
    elif config.metric == "identity":
        inv_metric = {"inv_metric": np.eye(len(param_names))}

    return model, data_path, seed, init, inv_metric


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


def run_sampler(model, data_path, seed, init, inv_metric, burn_in):
    nuts = model.sample(
        data=data_path,
        chains=1,
        seed=seed,
        inits=init,
        iter_sampling=int(wandb.config.gradient_budget / 5),
        metric=inv_metric,
        # b/c init from reference draw, no need for warm up
        adapt_init_phase=burn_in,
        show_console=False,
    )
    return nuts


def enforce_gradient_budget(nuts, config):
    nuts_np = nuts.draws(concat_chains=True)
    grad_evals = np.cumsum(nuts_np[:, 4])
    idx = np.argwhere(grad_evals > config.gradient_budget)[0][0]
    return nuts_np[:idx, :]


def format_history(cmdstan_history, config):
    nuts_np = enforce_gradient_budget(cmdstan_history, config)
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
    }
    
    # wandb logging
    log_freq, prev_log = max(1, config.gradient_budget // 10000), 0
    total_grads = history["grad_evals"][-1]
    for idx in range(len(history["grad_evals"])):
        grad = history["grad_evals"][idx]
        if grad - prev_log >= log_freq or grad == total_grads or prev_log == 0:
            log_history_to_wandb(history, idx)
            prev_log = grad
    
    return history


def main(config):
    if config.generate_history:
        model, data_path, seed, init, inv_metric = configure_sampler(config)
        cmdstan_history = run_sampler(model, data_path, seed, init, inv_metric, config.burn_in)
        # cmdstan's nuts implementation returns history in a different format
        history = format_history(cmdstan_history, config)
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
    parser = nuts_argument_parser()
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
    ]

    group = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and k not in ["chain"]
        ]
    )
    run_name = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams
        ]
    )

    WANDB_RUN_GROUP = group  # set environment var
    wandb.init(
        config=args,
        name=run_name,
        group=group,
        project=args.experiment,
        job_type=args.sampler_type,
        save_code=False,
    )
    main(wandb.config)
