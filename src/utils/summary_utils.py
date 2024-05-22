from csv import DictWriter
from functools import lru_cache
import logging
import os

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import polars as pl

from .save_utils import get_group


def _get_summary_data(config, metrics):
    group = get_group(config.sampler.params)
    history_path = os.path.join(
        HydraConfig.get().runtime.output_dir,
        f"history__chain={config.sampler.chain}.npz",
    )
    metric_path = os.path.join(
        HydraConfig.get().runtime.output_dir,
        f"metrics__chain={config.sampler.chain}.npz",
    )

    data = {
        "tags": config.wandb.tags,
        "error_param": metrics["error_param"][-1],
        "error_param_squared": metrics["error_param_squared"][-1],
        "chain": config.sampler.chain,
        "group": group,
        "history_path": history_path,
        "metric_path": metric_path,
    } | OmegaConf.to_container(config.sampler.params)

    if config.posterior.name == "funnel10":
        data.update(
            {
                "error_log_scale": metrics["error_log_scale"][-1],
                "error_log_scale_squared": metrics["error_log_scale_squared"][-1],
                "error_latent": metrics["error_latent"][-1],
                "error_latent_squared": metrics["error_latent_squared"][-1],
            }
        )

    return data


def write_summary(config, metrics):
    logging.debug("Writing summary")

    new_summary = not os.path.exists(config.summary.path)

    with open(config.summary.path, "a") as f:
        writer = DictWriter(f, fieldnames=config.summary.fieldnames)
        logging.debug(f"Writing to {config.summary.path}")
        if new_summary:
            writer.writeheader()
        logging.debug("Writing summary data")
        writer.writerow(_get_summary_data(config, metrics))
    logging.debug("Summary completed")


def my_load(path, arr_names, n_samples: int = None):
    res = list()

    with np.load(path, mmap_mode="r") as arr_dict:
        some_arr = arr_dict[arr_names[0]]
        some_shape = some_arr.shape
        step = max(1, some_shape[0] // n_samples) if n_samples else 1
        
        for arr_name in arr_names:
            arr = arr_dict[arr_name]
            arr = arr[::-1][::step][::-1]  # reverse to include last element
            res.append(arr)

    return np.column_stack(res)


def read_from_summary(
    summary, history_list: list = None, metrics_list: list = None, n_samples: int = None
):
    res = list()
    
    row_idx = [
        i
        for i, col in enumerate(summary.columns)
        if col not in history_list + metrics_list
    ]
    
    history_idx = summary.columns.index("history_path")
    metrics_idx = summary.columns.index("metric_path")

    for row in summary.iter_rows():
        history = my_load(row[history_idx], history_list, n_samples)
        metrics = my_load(row[metrics_idx], metrics_list, n_samples)
        row = [row[i] for i in row_idx] # remove duplicates
        rows = [list(row) for _ in range(metrics.shape[0])]

        res.extend(
            [r + list(history[i]) + list(metrics[i]) for i, r in enumerate(rows)]
        )

    if "draws" in history_list:
        num_params = np.shape(history)[1] - len(history_list) + 1
        replacement = [f"p{i}" for i in range(num_params)]
        idx = history_list.index("draws")
        history_list = history_list[:idx] + replacement + history_list[idx + 1 :]

    cols = [summary.columns[i] for i in row_idx] + history_list + metrics_list
    return pl.DataFrame(data=res, schema=cols, schema_overrides=summary.schema)
