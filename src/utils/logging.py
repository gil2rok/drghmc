import contextlib
import logging

from omegaconf import OmegaConf, open_dict
import wandb


def update_config_with_chain(config, chain):
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.sampler.chain = chain
    return config


def update_config_with_seed(config, seed):
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.sampler.seed = seed
    return config


def logging_context(config, job_type):
    if job_type == "history":
        logging.info(
            f"Generating history for chain {config.sampler.chain + 1}"
        )
    elif job_type == "metrics":
        logging.info(f"Generating metrics for chain {config.sampler.chain + 1}")

    log_history_bool = config.logging.log_history and job_type == "history"
    log_metrics_bool = config.logging.log_metrics and job_type == "metrics"

    if config.logging.logger == "wandb" and (log_history_bool or log_metrics_bool):
        wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        
        sampler_params = wandb_config["sampler"]["params"]
        irrelevant_keys = ["probabilistic", "metric", "adapt_metric"]
        group_name = "__".join(
            sorted(
                [f"{k}={v}"
                    for k, v in sampler_params.items()
                    if k not in irrelevant_keys and v is not None
                ]
            )
        )
        logging.info(f"Wandb group name:\t {group_name}")

        
        run = wandb.init(
            project=config.wandb.project,
            group=group_name,
            config=wandb_config,
            job_type=job_type,
            reinit=True,
            tags=[config.wandb.tags],
            settings=wandb.Settings(start_method="fork", _service_wait=300),
        )
    else:
        run = contextlib.nullcontext()
    return run


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
        wandb.define_metric(
            f"param_{i}/trace", step_metric="grad_evals", summary="mean"
        )


def log_history_to_wandb(history, idx=-1):
    cur_draw = history["draws"][idx]
    num_params = len(cur_draw)
    init_wandb(num_params)

    log_dict = {f"param_{i}/trace": param for i, param in enumerate(cur_draw)}
    log_dict["grad_evals"] = history["grad_evals"][idx]
    log_dict["acceptance_prob/acceptance_prob"] = history["accept_prob"][idx]
    wandb.log(log_dict)
