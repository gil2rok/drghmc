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
    """ Initialize sampler from reference draw
    
    Initialize each chain from the last draw of the reference chain.
    
    If there are not enough reference chains, use modulos to select 2nd to last draw, 
    3rd to last draw, etc. Assumes that the reference draws are thinned to be independent.
    
    Ensure array is contiguous b/c Stan models require it.
    """
    
    if posterior == "stochastic_volatility":
        fname = os.path.join(posterior_dir, posterior, "stochastic_volatility.inits.npy")
        return np.load(fname)
    
    ref_draws = get_ref_draws(posterior, posterior_dir)
    
    num_chains = ref_draws.shape[0]
    chain_idx = chain % num_chains
    draw_idx = -1 - (chain // num_chains)
        
    return np.ascontiguousarray(ref_draws[chain_idx, :, draw_idx])


def get_diagonal_covariance(posterior, posterior_dir):
    ref_draws = get_ref_draws(posterior, posterior_dir)
    return np.diag(np.var(ref_draws, axis=2).mean(axis=0))


def get_ref_draws_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    
    if posterior_paths["ref_draws_path"] is None:
        raise FileNotFoundError(f"Reference draws file not found for {posterior} posterior in {posterior_dir}.")
    
    return posterior_paths["ref_draws_path"]


def get_analytic_params_path(posterior, posterior_dir):
    posterior_paths = _get_posterior_paths(posterior, posterior_dir)
    return posterior_paths["analytic_params_path"]


@lru_cache(maxsize=10)
def get_analytic_params(posterior, posterior_dir):
    analytic_params_path = get_analytic_params_path(posterior, posterior_dir)
    
    analytic_params = dict(
        params_mean=None, 
        params_std=None, 
        params_squared_mean=None, 
        params_squared_std=None
    )
    
    try:
        with open(analytic_params_path, "r") as f:
            for k, v in json.load(f).items():
                analytic_params[k] = np.array(list(v.values()))
    except:
        logging.debug(f"Analytic parameters not found for {posterior} posterior.")
        
    return analytic_params


@lru_cache(maxsize=10)
def get_true_params_mean(posterior, posterior_dir):
    """Get the mean of posterior's true parameters, by reference draw estimation or analytically.
    
    Check for analytically known parameters. Otherwise, estimate the mean from 
    reference draws. If only some parameters are known analytically, estimate the 
    remaining parameters.
    
    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors
        
    Returns:
        true_params_mean: numpy array of true draws [num_params]
    """
    
    if posterior == "stochastic_volatility":
        fname = os.path.join(posterior_dir, posterior, "stochastic_volatility.true_params.npz")
        return np.load(fname)["mean"]
    
    ref_draws = get_ref_draws(posterior, posterior_dir)
    ref_params_mean = ref_draws.mean(axis=(0, 2))

    analytic_params = get_analytic_params(posterior, posterior_dir)
    analytic_params_mean = analytic_params["params_mean"]
    
    if analytic_params_mean is None:
        true_params_mean = ref_params_mean
        
    else:
        true_params_mean = np.empty_like(ref_params_mean)
        
        iterator = enumerate(zip(ref_params_mean, analytic_params_mean))
        for idx, (estimate, analytic) in iterator:
            
            if analytic is not None:
                true_params_mean[idx] = analytic
            else:
                true_params_mean[idx] = estimate
    
    return true_params_mean


@lru_cache(maxsize=10)
def get_true_params_std(posterior, posterior_dir):
    """Get a posterior's true parameters standard deviation by reference draw estimation or analytically.
    
    Check for analytically known parameters. Otherwise, estimate the standard deviation from reference draws. If only some parameters are known analytically, estimate the remaining parameters.
    
    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors
        
    Returns:
        true_params_std: numpy array of true draws standard deviation [num_params]
    """
    
    if posterior == "stochastic_volatility":
        fname = os.path.join(posterior_dir, posterior, "stochastic_volatility.true_params.npz")
        return np.load(fname)["std"]
    
    ref_draws = get_ref_draws(posterior, posterior_dir)
    ref_params_std = ref_draws.std(axis=(0, 2))

    analytic_params = get_analytic_params(posterior, posterior_dir)
    analytic_params_std = analytic_params["params_std"]
    
    if analytic_params_std is None:
        true_params_std = ref_params_std
    
    else:
        true_params_std = np.empty_like(ref_params_std)
        
        iterator = enumerate(zip(ref_params_std, analytic_params_std))
        for idx, (estimate, analytic) in iterator:
            
            if analytic is not None:
                true_params_std[idx] = analytic
            else:
                true_params_std[idx] = estimate
    
    return true_params_std


@lru_cache(maxsize=10)
def get_true_params_squared_mean(posterior, posterior_dir):
    """Get a posterior's true parameters squared by reference draw estimation or analytically.
    
    Check for analytically known parameters squared, equivalent to the variance. Otherwise, estimate the mean of reference draws squared. If only some parameters are known analytically, estimate the remaining parameters.
    
    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors
        
    Returns:
        true_params_squared_mean: numpy array of true draws squared [num_params]
    """
    if posterior == "stochastic_volatility":
        fname = os.path.join(posterior_dir, posterior, "stochastic_volatility.true_params.npz")
        return np.load(fname)["squard_mean"]
    
    ref_draws = get_ref_draws(posterior, posterior_dir)
    ref_draws_squared = np.square(ref_draws)
    ref_params_squared_mean = ref_draws_squared.mean(axis=(0, 2))

    analytic_params = get_analytic_params(posterior, posterior_dir)
    analytic_params_squared_mean = analytic_params["params_squared_mean"]
    
    if analytic_params_squared_mean is None:
        true_params_squared_mean = ref_params_squared_mean
    
    else:
        true_params_squared_mean = np.empty_like(ref_params_squared_mean)
        
        iterator = enumerate(zip(ref_params_squared_mean, analytic_params_squared_mean))
        for idx, (estimate, analytic) in iterator:
            
            if analytic is not None:
                true_params_squared_mean[idx] = analytic
            else:
                true_params_squared_mean[idx] = estimate
                
    return true_params_squared_mean


@lru_cache(maxsize=10)
def get_true_params_squared_std(posterior, posterior_dir):
    """Get a posterior's true parameters squared standard deviation by reference draw estimation or analytically.
    
    Check for analytically known parameters squared standard deviation, equivalent to the standard deviation. Otherwise, estimate the standard deviation of reference draws squared. If only some parameters are known analytically, estimate the remaining parameters.
    
    Args:
        posterior: name of the posterior
        posterior_dir: path to the directory containing posteriors
        
    Returns:
        true_params_squared_std: numpy array of true draws squared standard deviation [num_params]
    """
    if posterior == "stochastic_volatility":
        fname = os.path.join(posterior_dir, posterior, "stochastic_volatility.true_params.npz")
        return np.load(fname)["squared_std"]
    
    ref_draws = get_ref_draws(posterior, posterior_dir)
    ref_draws_squared = np.square(ref_draws)
    ref_params_squared_std = ref_draws_squared.std(axis=(0, 2))

    analytic_params = get_analytic_params(posterior, posterior_dir)
    analytic_params_squared_std = analytic_params["params_squared_std"]
    
    if analytic_params_squared_std is None:
        true_params_squared_std = ref_params_squared_std
    
    else:
        true_params_squared_std = np.empty_like(ref_params_squared_std)
        
        iterator = enumerate(zip(ref_params_squared_std, analytic_params_squared_std))
        for idx, (estimate, analytic) in iterator:
            
            if analytic is not None:
                true_params_squared_std[idx] = analytic
            else:
                true_params_squared_std[idx] = estimate
                
    return true_params_squared_std


@lru_cache(maxsize=1)
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

    logging.debug(f"Loading reference draws from {ref_draws_path}")
    
    logging.debug(f"Unzipping {ref_draws_path}.")
    ref_draws_dict = None
    with ZipFile(ref_draws_path + ".zip", "r") as z:
        try:
            fname = f"{posterior}.ref_draws.json"
            with z.open(fname, "r") as f:
                ref_draws_dict = json.load(f)
        except:
            fname = ref_draws_path.split("/")[-1]
            with z.open(fname, "r") as f:
                ref_draws_dict = json.load(f)
                
    logging.debug(f"Reference draws unzipped.")
            
    num_chains = len(ref_draws_dict)
    num_params = len(ref_draws_dict[0])
    num_draws = len(list(ref_draws_dict[0].values())[0])
    
    logging.debug(f"Converting reference draws to numpy array.")
    ref_draws = np.zeros((num_chains, num_params, num_draws))
    for chain_idx, chain in enumerate(ref_draws_dict):
        for param_idx, param in enumerate(chain.values()):
            ref_draws[chain_idx, param_idx, :] = np.array(param)
    
    return ref_draws


def get_param_names(posterior, posterior_dir):
    ref_draws_path = get_ref_draws_path(posterior, posterior_dir)

    ref_draws_dict = None
    with ZipFile(ref_draws_path + ".zip", "r") as z:
        try:
            fname = f"{posterior}.ref_draws.json"
            with z.open(fname, "r") as f:
                ref_draws_dict = json.load(f)
        except:
            fname = ref_draws_path.split("/")[-1]
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
        logging.debug(f"Data file not found for {posterior_name} posterior.")
    
    try:
        ref_draws_path = posterior.reference_draws_file_path()
        logging.debug(f"Reference draws file found for {posterior_name} posterior at {ref_draws_path}")
        logging.debug(f"Ref draws info: {posterior.reference_draws_info()}")
    except:
        ref_draws_path = None
        logging.debug(f"Reference draws file not found for {posterior_name} posterior.")
        
    posterior_paths = dict(
        model_path=model_path,
        data_path=data_path,
        ref_draws_path=ref_draws_path,
        analytic_params_path=None, # not included in PosteriorDB
    )
    return posterior_paths


def _get_custom_posterior(posterior, posterior_dir):
    path = os.path.join(posterior_dir, posterior)
    
    model_path = os.path.join(path, f"{posterior}.stan")
    
    data_path = os.path.join(path, f"{posterior}.data.json")
    if not os.path.isfile(data_path):
        data_path = None
        logging.debug(f"Data file not found for {posterior}  posterior.")
    
    ref_draws_path = os.path.join(path, f"{posterior}.ref_draws.json")
    if not os.path.isfile(ref_draws_path + ".zip"):
        ref_draws_path = None
        logging.debug(f"Reference draws file not found for {posterior} posterior.")
        
    analytic_params_path = os.path.join(path, f"{posterior}.analytic_params.json")
    if not os.path.isfile(analytic_params_path):
        analytic_params_path = None
        logging.debug(f"Analytic parameters file not found for {posterior} posterior.")
        
    posterior_paths = dict(
        model_path=model_path,
        data_path=data_path,
        ref_draws_path=ref_draws_path,
        analytic_params_path=analytic_params_path,
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
    