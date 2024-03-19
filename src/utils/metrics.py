from collections import defaultdict
import logging

import numpy as np
import wandb

from .posteriors import get_true_params_mean, get_true_params_squared_mean, get_true_params_std, get_true_params_squared_std


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
    num_params = len(metrics["se1"][idx])

    log_dict = {
        "grad_evals": history["grad_evals"][idx],
        # round grad evals to nearest thousand for accurate plotting
        # "grad_evals": np.round(history["grad_evals"][idx], decimals=-3),
        "squared_error/se1": metrics["max_se1"][idx],
        "squared_error/se2": metrics["max_se2"][idx],
    }
    
    if len(metrics["max_rse1"]) > 0:
        log_dict["squared_error/rse1"] = metrics["max_rse1"][idx]
    if len(metrics["max_rse2"]) > 0:
        log_dict["squared_error/rse2"] = metrics["max_rse2"][idx]
    
    for i in range(num_params):
        log_dict[f"param_{i}/se1"] = metrics["se1"][idx][i]
        log_dict[f"param_{i}/se2"] = metrics["se2"][idx][i]
        if len(metrics["rse1"]) > 0:
            log_dict[f"param_{i}/rse1"] = metrics["rse1"][idx][i]
        if len(metrics["rse2"]) > 0:
            log_dict[f"param_{i}/rse2"] = metrics["rse2"][idx][i]
    
    try:
        log_dict["funnel10_squared_error/se1_scale"] = metrics["se1_scale"][idx]
        log_dict["funnel10_squared_error/se1_latent"] = metrics["se1_latent"][idx]
        log_dict["funnel10_squared_error/se2_scale"] = metrics["se2_scale"][idx]
        log_dict["funnel10_squared_error/se2_latent"] = metrics["se2_latent"][idx]
        
        if len(metrics["rse1_scale"]) > 0:
            log_dict["funnel10_squared_error/rse1_scale"] = metrics["rse1_scale"][idx]
            log_dict["funnel10_squared_error/rse1_latent"] = metrics["rse1_latent"][idx]
            
        if len(metrics["rse2_scale"]) > 0:
            log_dict["funnel10_squared_error/rse2_scale"] = metrics["rse2_scale"][idx]
            log_dict["funnel10_squared_error/rse2_latent"] = metrics["rse2_latent"][idx]
    except:
        pass
    
    wandb.log(log_dict)


def _log_wandb_final(config, history, metrics):
    log_dict = dict()
    num_draws = metrics["se1"].shape[0]
    log_freq = max(1, num_draws // config.wandb.points_per_metric)
    
    # if history["acceptance"] is not None:
    #     acceptance_table = wandb.Table(
    #         data=[[val] for val in history["acceptance"][::log_freq]],
    #         columns=["acceptance"], 
    #     )
    #     log_dict["acceptance/acceptance"] = acceptance_table
    
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


def log_metrics_to_wandb(config, history, metrics):
    num_draws, num_params = metrics["se1"].shape
    _init_wandb(num_params)
    
    gradient_budget = config.sampler.gradient_budget
    points_per_metric = config.wandb.points_per_metric
    log_freq, prev_log = max(1, gradient_budget // points_per_metric), 0
    total_grads = history["grad_evals"][-1]
    
    for idx in range(len(history["grad_evals"])):
        grad = history["grad_evals"][idx]
        
        log_iter_bool = (
            (grad - prev_log >= log_freq) or (prev_log == 0) or (grad == total_grads)
        )
        if log_iter_bool:
            _log_wandb_iter(history, metrics, idx)
            prev_log = grad
        
    _log_wandb_iter(history, metrics, -1)
    _log_wandb_final(config, history, metrics)


def compute_jump_distance(draws):
    jump_dist = np.linalg.norm(np.diff(draws, axis=0), axis=1) ** 2
    return list(jump_dist)


def compute_uturn_distance(draws):
    diff = np.diff(draws, axis=0)
    
    sign_flips = np.where(np.diff(np.sign(diff), axis=0) != 0)
    sign_flips = np.hstack((np.array([0]), sign_flips[0], np.array([len(draws)])))
    
    uturn_dist = list()
    for i, j in zip(sign_flips[:-1], sign_flips[1:]):
        arr = np.linalg.norm(diff[i:j], axis=1) ** 2
        uturn_dist += list(arr)
    
    return uturn_dist


def _merge_chains_by_grad_evals(history_list):
    merged_list = list()
    for history in history_list:
        grad_evals = history["grad_evals"]
        draws = history["draws"]
        diff = np.hstack((grad_evals[0], np.diff(grad_evals)))
        merged_list.append(np.hstack(
            (grad_evals.reshape(-1, 1), diff.reshape(-1, 1), draws)
        ))
        
    merged = np.vstack(merged_list)
    merged = merged[merged[:, 0].argsort()] # sort by grad evals
    cum_grad_evals = np.cumsum(merged[:, 1]) # cumulative grad evals
    merged = merged[:, 2:] # remove grad evals and diff
    return cum_grad_evals, merged


def compute_squared_error_variants(draws, grad_evals, config):
    metrics = dict()
    draws_squared = np.square(draws)
    
    denom = np.arange(1, len(draws) + 1)[:, None]
    est_params = np.cumsum(draws, axis=0) / denom
    est_params_squared = np.cumsum(draws_squared, axis=0) / denom
    
    true_params = get_true_params_mean(config.posterior.name, config.posterior.dir)
    true_params_squared = get_true_params_squared_mean(config.posterior.name, config.posterior.dir)
    
    true_params_std = get_true_params_std(config.posterior.name, config.posterior.dir)
    true_params_squared_std = get_true_params_squared_std(config.posterior.name, config.posterior.dir)
    
    logging.info(f"true_params:\t{true_params}")
    logging.info(f"est_params:\t{est_params[-1]}")
    logging.info(f"true_params_squared:\t{true_params_squared}")
    logging.info(f"est_params_squared:\t{est_params_squared[-1]}")
    logging.info(f"true_params_std:\t{true_params_std}")
    logging.info(f"true_params_squared_std:\t{true_params_squared_std}")
    
    metrics["se1"] = np.square(est_params - true_params)
    metrics["se2"] = np.square(est_params_squared - true_params_squared)
    metrics["max_se1"] = np.max(metrics["se1"], axis=1)
    metrics["max_se2"] = np.max(metrics["se2"], axis=1)
    
    metrics["c1"] = np.max(
        np.square(est_params - true_params) / np.square(true_params_std),
        axis=1
    )
    metrics["c2"] = np.max(
        np.square(est_params_squared - true_params_squared) / np.square(true_params_squared_std), 
        axis=1
    )
    
    metrics["c1_log_scale"] = (np.square(est_params - true_params) / np.square(true_params_std))[:, 0]
    metrics["c2_log_scale"] = (np.square(est_params_squared - true_params_squared) / np.square(true_params_squared_std))[:, 0]
    metrics["c1_latent"] = np.max(
        (np.square(est_params - true_params) / np.square(true_params_std))[:, 1:],
        axis=1
    )
    metrics["c2_latent"] = np.max(
        (np.square(est_params_squared - true_params_squared) / np.square(true_params_squared_std))[:, 1:],
        axis=1
    )
    
    
    
    if np.all(true_params != 0):
        metrics["rse1"] = np.square((est_params - true_params) / true_params)
        metrics["max_rse1"] = np.max(metrics["rse1"], axis=1)
        
    if np.all(true_params_squared != 0):
        metrics["rse2"] = np.square((est_params_squared - true_params_squared) / true_params_squared)
        metrics["max_rse2"] = np.max(metrics["rse2"], axis=1)
        
    # specific metrics for funnel10 posterior
    if config.posterior.name == "funnel10":
        metrics["se1_scale"] = metrics["se1"][:, 0]
        metrics["se1_latent"] = np.max(metrics["se1"][:, 1:], axis=1)
        
        metrics["se2_scale"] = metrics["se2"][:, 0]
        metrics["se2_latent"] = np.max(metrics["se2"][:, 1:], axis=1)
        
        if np.all(true_params != 0):
            metrics["rse1_scale"] = metrics["rse1"][:, 0]
            metrics["rse1_latent"] = np.max(metrics["rse1"][:, 1:], axis=1)
            
        if np.all(true_params_squared != 0):
            metrics["rse2_scale"] = metrics["rse2"][:, 0]
            metrics["rse2_latent"] = np.max(metrics["rse2"][:, 1:], axis=1)
        
    return metrics


def compute_metrics(history, config):
    metrics = defaultdict(list)
    
    metrics["jump_dist"] = compute_jump_distance(history["draws"])
    metrics["uturn_dist"] = compute_uturn_distance(history["draws"])
    metrics.update(
        compute_squared_error_variants(history["draws"], history["grad_evals"], config)
    )
    
    if config.logging.log_metrics and config.logging.logger == "wandb":
        log_metrics_to_wandb(config, history, metrics)

    return metrics