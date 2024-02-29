from collections import defaultdict
import time

import numpy as np
import wandb

from utils.save_utils import get_data_dir
from utils.posteriors import processed_ref_draws
from utils.posteriors_new import get_true_params, get_true_params_squared


def _streaming_mean(x, n, mean):
    """Efficiently compute the mean of a stream of numbers.

    Args:
        x (float): The new value to be added to the mean.
        n (int): The number of values in the stream.
        mean (float): The current mean of the stream.

    Returns:
        float: The updated mean of the stream.
    """
    return mean + (x - mean) / n


def _init_wandb(num_params):
    # define wandb metrics
    wandb.define_metric("grad_evals", summary="max")
    wandb.define_metric("max_se1", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_se2", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_rse1", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_rse2", step_metric="grad_evals", summary="last")
    
    for i in range(num_params):
        wandb.define_metric(f"param_{i}/se1", step_metric="grad_evals", summary="last")
        wandb.define_metric(f"param_{i}/se2", step_metric="grad_evals", summary="last")
        wandb.define_metric(f"param_{i}/rse1", step_metric="grad_evals", summary="last")
        wandb.define_metric(f"param_{i}/rse2", step_metric="grad_evals", summary="last")


def _log_wandb_iter(history, metrics, idx):
    num_params = len(history["draws"][-1])

    log_dict = {
        "grad_evals": history["grad_evals"][idx],
        "squared_error/se1": metrics["max_se1"][idx],
        "squared_error/se2": metrics["max_se2"][idx],
    }
    if metrics["max_rse1"]:
        log_dict["squared_error/rse1"] = metrics["max_rse1"][idx]
    if metrics["max_rse2"]:
        log_dict["squared_error/rse2"] = metrics["max_rse2"][idx]
    
    for i in range(num_params):
        log_dict[f"param_{i}/se1"] = metrics["se1"][idx][i]
        log_dict[f"param_{i}/se2"] = metrics["se2"][idx][i]
        if metrics["rse1"]:
            log_dict[f"param_{i}/rse1"] = metrics["rse1"][idx][i]
        if metrics["rse2"]:
            log_dict[f"param_{i}/rse2"] = metrics["rse2"][idx][i]
    wandb.log(log_dict)


def _log_wandb_final(config, history, metrics):
    log_dict = dict()
    log_freq = max(1, config.gradient_budget // 1000)
    
    if history["acceptance"] is not None:
        acceptance_table = wandb.Table(
            data=[[val] for val in history["acceptance"][::log_freq]],
            columns=["acceptance"], 
        )
        log_dict["acceptance/acceptance"] = acceptance_table
    
    if metrics["jump_dist"]:
        jump_dist_table = wandb.Table(
            data=[[val] for val in metrics["jump_dist"][::log_freq]],
            columns=["jump_dist"],
        )
        log_dict["jump_dist/jump_dist"] = jump_dist_table
        
    if metrics["uturn_dist"]:
        uturn_dist_table = wandb.Table(
            data=[[val] for val in metrics["uturn_dist"][::log_freq]],
            columns=["uturn_dist"],
        )
        log_dict["uturn_dist/uturn_dist"] = uturn_dist_table
        
    wandb.log(log_dict)


def compute_metrics(history, config):
    # initialize defaults
    metrics = defaultdict(list)
    _init_wandb(len(history["draws"][0]))
    
    prev_draw = None
    true_params = get_true_params(config.posterior, config.posterior_dir)
    true_params_squared = get_true_params_squared(config.posterior, config.posterior_dir)
    est_params, est_params_squared = 0, 0
    uturn_dict = defaultdict(int)
    log_freq, prev_log = max(1, config.gradient_budget // 100), 0
    
    for idx, draw in enumerate(history["draws"]):
        # jump distance
        if prev_draw is not None:
            metrics["jump_dist"] += [np.linalg.norm(draw - prev_draw) ** 2]
        prev_draw = draw
        
        # update estimated parameters with streaming mean
        est_params = _streaming_mean(draw, idx + 1, est_params)
        est_params_squared = _streaming_mean(np.square(draw), idx + 1, est_params_squared)
        
        # squared error
        metrics["se1"] += [np.square(est_params - true_params)]
        metrics["se2"] += [np.square(est_params_squared - true_params_squared)]
        metrics["max_se1"] += [np.max(metrics["se1"][-1])]
        metrics["max_se2"] += [np.max(metrics["se2"][-1])]
        
        # relative squared error
        if np.all(true_params != 0): # check for division by zero
            metrics["rse1"] += [np.square((est_params - true_params) / true_params)]
            metrics["max_rse1"] += [np.max(metrics["rse1"][-1])]

        if np.all(true_params_squared != 0): # check for division by zero
            metrics["rse2"] += [np.square((est_params_squared - true_params_squared) / true_params_squared)]
            metrics["max_rse2"] += [np.max(metrics["rse2"][-1])]
    
        # u-turn distance
        for old_draw, old_dist in tuple(uturn_dict.items()):
            dist = np.linalg.norm(draw - old_draw) # squared distance 
            if dist < old_dist:
                metrics["uturn_dist"] += [dist]
                del uturn_dict[old_draw]
            elif dist >= old_dist:
                uturn_dict[old_draw] = dist
        uturn_dict[tuple(draw)] = 0.0
        
        # log to wandb (ensure record first and last draw)    
        if history["grad_evals"][idx] - prev_log >= log_freq or idx == 0 or idx == len(history["draws"]) - 1:
            _log_wandb_iter(history, metrics, idx)
            prev_log = idx
    _log_wandb_final(config, history, metrics)
    return metrics