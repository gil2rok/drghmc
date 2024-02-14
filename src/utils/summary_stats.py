import json
import logging
import os
from functools import lru_cache
from zipfile import ZipFile

import numpy as np
from posteriordb import PosteriorDatabase

def get_ref_draws(posterior_path, posterior_name):
    """Returns list of dictionaries, where each dict represents an individual chain.
    Each dict has keys as parameter names and values as a list of parameter draws.
    """
    try:  # try to load posterior from PDB
        path = os.path.join(posterior_path, "posteriordb/posterior_database")
        pdb = PosteriorDatabase(path)
        posterior = pdb.posterior(posterior_name)
        ref_draws = posterior.reference_draws()

    except:  #  load posterior from custom model
        path = os.path.join(posterior_path, posterior_name)
        ref_draws_path = os.path.join(path, f"{posterior_name}.ref_draws.json.zip")
        ref_draws = json.loads(
            ZipFile(ref_draws_path)
            .read(f"{posterior_name}.ref_draws.json")
            .decode("utf-8")
        )

    return ref_draws

def standard_error_old(draws, ref_draws):
    avg_ref_draws = np.mean(ref_draws, axis=(0,2)) # [num_params]
    standard_error = np.square(draws - avg_ref_draws)
    return np.mean(standard_error, axis=0).tolist()


def squared_error(draws, ref_draws):
    avg_ref_draws = np.mean(ref_draws, axis=(0,2)) # [num_params]
    avg_draws = np.mean(draws, axis=0) # [num_params]
    squared_error = np.square(avg_draws - avg_ref_draws).tolist()
    
    logging.info(f"avg_ref_draws: {avg_ref_draws}")
    logging.info(f"new_draw: {draws[-1]} with shape {draws.shape}")
    logging.info(f"avg_draws: {avg_draws}")
    logging.info(f"squared_error: {squared_error}\n")
    
    return squared_error


def relative_squared_error(draws, ref_draws):
    avg_ref_draws = np.mean(ref_draws, axis=(0,2)) # [num_params]
    avg_draws = np.mean(draws, axis=0) # [num_params]
    return np.square((avg_draws - avg_ref_draws) / avg_ref_draws).tolist()


def streaming_avg(new_draw, avg_draws, counter):
    draws_sum = avg_draws * counter + new_draw
    return draws_sum / (counter + 1)


def relative_squared_error2(avg_draws, avg_ref_draws):
    return np.square((avg_draws - avg_ref_draws) / avg_ref_draws).tolist()


def squared_error2(avg_draws, avg_ref_draws):
    return np.square(avg_draws - avg_ref_draws).tolist()
    
    
def streaming_se(new_draw, avg_draws, avg_ref_draws, counter):
    draws_sum = avg_draws * counter + new_draw
    new_avg_draws = draws_sum / (counter + 1)
    
    squared_error = np.square(new_avg_draws - avg_ref_draws)
    return squared_error, new_avg_draws


def streaming_relative_se(new_draw, avg_draws, avg_ref_draws, counter, sampler):
    draws_sum = avg_draws * counter + new_draw
    new_avg_draws = draws_sum / (counter + 1)

    relative_squared_error = np.square((new_avg_draws - avg_ref_draws) / avg_ref_draws)
    
    logging.info(f"gradient evals: {sampler._model.log_density_gradient.calls}\tdraw number: {counter}")
    logging.info(f"avg_ref_draws: {avg_ref_draws}")
    logging.info(f"new_avg_draws: {new_avg_draws}")
    logging.info(f"relative_squared_error: {relative_squared_error}\n")
    return relative_squared_error, new_avg_draws


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
    }
    return summary_stats
