from functools import lru_cache

import numpy as np

from src.process_samples import get_ref_draws


def standard_error_old(draws, ref_draws):
    avg_ref_draws = np.mean(ref_draws, axis=(0,2)) # [num_params]
    standard_error = np.square(draws - avg_ref_draws)
    return np.mean(standard_error, axis=0).tolist()


def squared_error(draws, ref_draws):
    avg_ref_draws = np.mean(ref_draws, axis=(0,2)) # [num_params]
    avg_draws = np.mean(draws, axis=0) # [num_params]
    return np.square(avg_draws - avg_ref_draws).tolist()


def ess(draws, ref_draws):
    avg_ref_draws = np.mean(ref_draws, axis=(0,2))
    standard_error = draws - avg_ref_draws
    
    standard_deviation = np.std(draws, axis=0)
    return np.square(standard_deviation / standard_error).tolist()


@lru_cache(maxsize=1)
def processed_ref_draws(posterior_path, posterior_name):
    ref_draws_dict = get_ref_draws(posterior_path, posterior_name)
        
    num_chains = len(ref_draws_dict)
    num_params = len(ref_draws_dict[0])
    num_draws = len(list(ref_draws_dict[0].values())[0])
        
    ref_draws = np.zeros((num_chains, num_params, num_draws))
    for chain_idx, chain in enumerate(ref_draws_dict):
        for param_idx, param in enumerate(chain.values()):
            ref_draws[chain_idx, param_idx, :] = np.array(param)
            
    return ref_draws


def reach_tail(draws):
    num_reached_tail = np.sum(draws[:, 0] < -5)
    num_draws = draws.shape[0]
    return num_reached_tail / num_draws


def get_summary_stats(draws, hp):
    ref_draws = processed_ref_draws(hp.posterior_dir, hp.posterior)
    
    summary_stats = {
        "se1": squared_error(draws, ref_draws), 
        "se2": squared_error(np.square(draws), np.square(ref_draws)),
        "tail": reach_tail(draws),
    }
    return summary_stats
