from csv import DictWriter
import logging
import os

from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from .save_utils import get_group


def _get_summary_data(config, metrics):
    group = get_group(config.sampler.params)
    history_path = os.path.join(
        HydraConfig.get().runtime.output_dir,
        f"history_chain={config.sampler.chain}.npz"
    )
    metric_path = os.path.join(
        HydraConfig.get().runtime.output_dir,
        f"metrics_chain={config.sampler.chain}.npz"
    )
    
    data = {
        "tags": config.wandb.tags,
        "c1": metrics["c1"][-1],
        "c2": metrics["c2"][-1],
        "c1_log_scale": metrics["c1_log_scale"][-1],
        "c2_log_scale": metrics["c2_log_scale"][-1],
        "c1_latent": metrics["c1_latent"][-1],
        "c2_latent": metrics["c2_latent"][-1],
        "se1_max": metrics["max_se1"][-1], 
        "se2_max": metrics["max_se2"][-1],
        "chain": config.sampler.chain,
        "group": group, 
        "history_path": history_path, 
        "metric_path": metric_path,
    } | OmegaConf.to_container(config.sampler.params)
    
    return data


def write_summary(config, metrics):
    logging.info("Writing summary")
    
    new_summary = not os.path.exists(config.summary.path)
    
    with open(config.summary.path, "a") as f:
        writer = DictWriter(f, fieldnames=config.summary.fieldnames)        
        logging.info(f"Writing to {config.summary.path}")
        if new_summary:
            writer.writeheader()
        logging.info("Writing summary data")
        writer.writerow(_get_summary_data(config, metrics)) 
