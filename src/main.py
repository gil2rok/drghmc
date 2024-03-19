import contextlib

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

from drghmc_runner import run_drghmc
from drhmc_runner import run_drhmc
from nuts_runner import run_nuts


def _init_logging(config: DictConfig, job_type: str) -> None:
    log_history_bool = config.logging.log_history and job_type == "history"
    log_metrics_bool = config.logging.log_metrics and job_type == "metrics"
    
    if config.logging.logger == "wandb" and (log_history_bool or log_metrics_bool):
        wandb.config = OmegaConf.to_container(
            config, 
            resolve=True, 
            throw_on_missing=True
        )
        run = wandb.init(
            project=config.wandb.project,
            group=None,
            config=wandb.config,
            job_type=job_type,
            reinit=True
        )
    else:
        run = contextlib.nullcontext()
    return run


def _update_config_with_chain(config: DictConfig, chain: int) -> DictConfig:
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.sampler.history.chain = chain
    return config


def generate_history(config: DictConfig) -> None:
    for chain in range(config.sampler.history.chains):
        config = _update_config_with_chain(config, chain)
        
        run = _init_logging(config, job_type="history")
        with run:
            
            if config.sampler.sampler_type == "drghmc":
                run_drghmc(config)
            elif config.sampler.sampler_type == "drhmc":
                run_drhmc(config)
            elif config.sampler.sampler_type == "nuts":
                run_nuts(config)


def generate_metrics(config: DictConfig) -> None:
    run = _init_logging(config, job_type="metrics")
    with run:
        pass


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if config.sampler.generate_history:
        generate_history(config)

    if config.sampler.generate_metrics:
        generate_metrics(config)

if __name__ == "__main__":
    main()