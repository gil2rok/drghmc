from collections import defaultdict
import logging

import numpy as np

from .posteriors import (
    get_true_params_mean,
    get_true_params_squared_mean,
    get_true_params_std,
    get_true_params_squared_std,
)


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


def compute_cost(draws, grad_evals, config):
    metrics = dict()
    draws_squared = np.square(draws)

    # compute estimated and true parameter statistics
    denom = np.arange(1, len(draws) + 1)[:, None]
    est_params_mean = np.cumsum(draws, axis=0) / denom
    est_params_squared_mean = np.cumsum(draws_squared, axis=0) / denom

    true_params_mean = get_true_params_mean(config.posterior.name, config.posterior.dir)
    true_params_squared_mean = get_true_params_squared_mean(
        config.posterior.name, config.posterior.dir
    )

    true_params_std = get_true_params_std(config.posterior.name, config.posterior.dir)
    true_params_squared_std = get_true_params_squared_std(
        config.posterior.name, config.posterior.dir
    )
    
    # compute cost metrics
    cost_param = np.abs(true_params_mean - est_params_mean) / true_params_std
    cost_param_squared = np.abs(
        true_params_squared_mean - est_params_squared_mean
    ) / true_params_squared_std
    
    metrics["error_param"] = np.max(cost_param, axis=1)
    metrics["error_param_squared"] = np.max(cost_param_squared, axis=1)
    
    # specific cost metrics for funnel10
    if config.posterior.name == "funnel10" or config.posterior.name == "funnel50":
        metrics["error_log_scale"] = cost_param[:, 0]
        metrics["error_log_scale_squared"] = cost_param_squared[:, 0]
        
        metrics["error_latent"] = np.max(cost_param[:, 1:], axis=1)
        metrics["error_latent_squared"] = np.max(cost_param_squared[:, 1:], axis=1)

    return metrics


def compute_metrics(history, config):
    metrics = defaultdict(list)

    metrics["jump_dist"] = compute_jump_distance(history["draws"])
    metrics["uturn_dist"] = compute_uturn_distance(history["draws"])
    metrics.update(
        compute_cost(history["draws"], history["grad_evals"], config)
    )

    return metrics
