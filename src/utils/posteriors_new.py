from functools import lru_cache
import json
import logging
import os
from zipfile import ZipFile

import numpy as np
from posteriordb import PosteriorDatabase


def get_model_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    return posterior_paths["model_path"]


def get_data_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    return posterior_paths["data_path"]


def get_init(posterior, posterior_dir, chain):
    ref_draws = get_ref_draws(posterior, posterior_dir)
    return np.ascontiguousarray(ref_draws[chain, :, 0])


def get_diagonal_covariance(posterior, posterior_dir):
    ref_draws = get_ref_draws(posterior, posterior_dir)
    return np.diag(np.var(ref_draws, axis=2).mean(axis=0))


def get_ref_draws_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    
    if posterior_paths["ref_draws_path"] is None:
        raise FileNotFoundError(f"Reference draws file not found for {posterior} in {posterior_dir}.")
    
    return posterior_paths["ref_draws_path"]


def get_analytic_params_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    return posterior_paths["analytic_params_path"]


def get_analytic_params_squared_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    return posterior_paths["analytic_params_squared_path"]


def get_analytic_params(posterior, posterior_dir):
    analytic_params_path = get_analytic_params_path(posterior, posterior_dir)
    
    try:
        with open(analytic_params_path, "r") as f:
            analytic_params = np.array(list(json.load(f).values()))
    except:
        analytic_params = None
        logging.info(f"Analytic parameters not found for {posterior}.")
        
    return analytic_params


def get_analytic_params_squared(posterior, posterior_dir):
    analytic_params_squared_path = get_analytic_params_squared_path(posterior, posterior_dir)
    
    try:
        with open(analytic_params_squared_path, "r") as f:
            analytic_params_squared= np.array(list(json.load(f).values()))
    except:
        analytic_params_squared = None
        logging.info(f"Analytic parameters squared not found for {posterior}.")
    
    return analytic_params_squared

@lru_cache(maxsize=10)
def get_true_params(posterior, posterior_dir):
    """Get a posterior's true parameters by reference draw estimation or analytically.
    
    Check for analytically known parameters. Otherwise, estimate the mean from 
    reference draws. If only some parameters are known analytically, estimate the 
    remaining parameters.
    
    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors
        
    Returns:
        true_draws: numpy array of true draws [num_params]
    """
    
    ref_draws = get_ref_draws(posterior, posterior_dir)
    estimated_params = ref_draws.mean(axis=(0, 2))

    analytic_params = get_analytic_params(posterior, posterior_dir)
    
    if analytic_params is None:
        true_draws = estimated_params
        
    else:
        true_draws = np.empty_like(estimated_params)
        
        iterator = enumerate(zip(estimated_params, analytic_params))
        for idx, (estimate, analytic) in iterator:
            
            if analytic is not None:
                true_draws[idx] = analytic
            else:
                true_draws[idx] = estimate
    
    return true_draws


@lru_cache(maxsize=10)
def get_true_params_squared(posterior, posterior_dir):
    """Get a posterior's true parameters squared by reference draw estimation or analytically.
    
    Check for analytically known parameters squared, equivalent to the variance. Otherwise, estimate the mean of reference draws squared. If only some parameters are known analytically, estimate the remaining parameters.
    
    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors
        
    Returns:
        true_draws: numpy array of true draws squared [num_params]
    """
        
    ref_draws = get_ref_draws(posterior, posterior_dir)
    ref_draws_squared = np.square(ref_draws)
    estimated_params_squared = ref_draws_squared.mean(axis=(0, 2))

    analytic_params_squared = get_analytic_params_squared(posterior, posterior_dir)
    
    if analytic_params_squared is None:
        true_draws_squared = estimated_params_squared
    
    else:
        true_draws_squared = np.empty_like(estimated_params_squared)
        
        iterator = enumerate(zip(estimated_params_squared, analytic_params_squared))
        for idx, (estimate, analytic) in iterator:
            
            if analytic is not None:
                true_draws_squared[idx] = analytic
            else:
                true_draws_squared[idx] = estimate
                
    return true_draws_squared


def get_ref_draws(posterior, posterior_dir):
    """Get a posterior's reference draws from the posterior directory.
    
    Get the reference draws path, unzip the file, load as json, and convert to numpy 
    array.

    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors

    Returns:
        ref_draws: numpy array of reference draws [num_chains, num_params, num_draws]
    """
    ref_draws_path = get_ref_draws_path(posterior, posterior_dir)

    ref_draws_dict = None
    with ZipFile(ref_draws_path + ".zip", "r") as z:
        fname = f"{posterior}.ref_draws.json"
        with z.open(fname, "r") as f:
            ref_draws_dict = json.load(f)
            
    num_chains = len(ref_draws_dict)
    num_params = len(ref_draws_dict[0])
    num_draws = len(list(ref_draws_dict[0].values())[0])
        
    ref_draws = np.zeros((num_chains, num_params, num_draws))
    for chain_idx, chain in enumerate(ref_draws_dict):
        for param_idx, param in enumerate(chain.values()):
            ref_draws[chain_idx, param_idx, :] = np.array(param)
    
    return ref_draws


def get_param_names(posterior, posterior_dir):
    ref_draws_path = get_ref_draws_path(posterior, posterior_dir)

    ref_draws_dict = None
    with ZipFile(ref_draws_path + ".zip", "r") as z:
        fname = f"{posterior}.ref_draws.json"
        with z.open(fname, "r") as f:
            ref_draws_dict = json.load(f)
            
    return list(ref_draws_dict[0].keys())


def _get_posteriordb_posterior(posterior_name, posterior_dir):
    path = os.path.join(posterior_dir, "posteriordb/posterior_database/")
    pdb = PosteriorDatabase(path)
    posterior = pdb.posterior(posterior_name)
    
    model_path = posterior.model.code_file_path("stan")
    
    try:
        data_path = posterior.data.file_path() # unzip data into tmp dir
        os.rename(data_path, data_path + ".json") # add .json extension for bridgestan
        data_path += ".json"
    except Exception as e:
        data_path = None
        logging.info(f"Data file not found for {posterior_name}.")
    
    try:
        ref_draws_path = posterior.reference_draws_file_path()
    except:
        ref_draws_path = None
        logging.info(f"Reference draws file not found for {posterior_name}.")
        
    posterior_paths = dict(
        model_path=model_path,
        data_path=data_path,
        ref_draws_path=ref_draws_path,
        analytic_params_path=None, # not included in PosteriorDB
        analytic_params_squared_path=None, # not included in PosteriorDB
    )
    return posterior_paths


def _get_custom_posterior(posterior, posterior_dir):
    path = os.path.join(posterior_dir, posterior)
    
    model_path = os.path.join(path, f"{posterior}.stan")
    
    data_path = os.path.join(path, f"{posterior}.data.json")
    if not os.path.isfile(data_path):
        data_path = None
        logging.info(f"Data file not found for {posterior}.")
    
    ref_draws_path = os.path.join(path, f"{posterior}.ref_draws.json")
    if not os.path.isfile(ref_draws_path + ".zip"):
        ref_draws_path = None
        logging.info(f"Reference draws file not found for {posterior}.")
        
    analytic_params_path = os.path.join(path, f"{posterior}.analytic_params.json")
    if not os.path.isfile(analytic_params_path):
        analytic_params_path = None
        logging.info(f"Analytic parameters file not found for {posterior}.")
        
    analytic_params_squared_path = os.path.join(path, f"{posterior}.analytic_params_squared.json")
    if not os.path.isfile(analytic_params_squared_path):
        analytic_params_squared_path = None
        logging.info(f"Analytic parameters squared file not found for {posterior}.")
        
    posterior_paths = dict(
        model_path=model_path,
        data_path=data_path,
        ref_draws_path=ref_draws_path,
        analytic_params_path=analytic_params_path,
        analytic_params_squared_path=analytic_params_squared_path,
    )
    return posterior_paths


@lru_cache(maxsize=10)
def _get_posterior_paths(posterior, posterior_dir):
    # try to load posterior from PosteriorDB library
    try:
        posterior_paths = _get_posteriordb_posterior(posterior, posterior_dir)
    except:
        # try to load posterior from custom model
        try:
            posterior_paths = _get_custom_posterior(posterior, posterior_dir)
        # if neither works, raise an error
        except:
            raise FileNotFoundError(f"Posterior {posterior} not found in {posterior_dir}.")
    
    return posterior_paths
    