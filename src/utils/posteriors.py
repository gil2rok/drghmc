import json, os
from zipfile import ZipFile
from functools import lru_cache

import bridgestan as bs
from cmdstanpy import CmdStanModel
import numpy as np
from posteriordb import PosteriorDatabase

from .models import BayesKitModel


def bayes_kit_posterior(model_name, posterior_path):
    try:  # try to load posterior from PDB
        posterior_origin = "pdb"
        path = os.path.join(posterior_path, "posteriordb/posterior_database")
        
        pdb = PosteriorDatabase(path)
        posterior = pdb.posterior(model_name)
        
        model_path = posterior.model.code_file_path("stan")
        data = posterior.data.values()
        ref_draws = posterior.reference_draws()
        
    except:  # load posterior from custom model
        posterior_origin = "custom"
        path = os.path.join(posterior_path, model_name)
        
        model_path = os.path.join(path, f"{model_name}.stan")
        
        data_path = os.path.join(path, f"{model_name}.data.json")
        data = json.load(open(data_path, "r"))
        
        ref_draws_path = os.path.join(path, f"{model_name}.ref_draws.json.zip")
        ref_draws = json.load(ZipFile(ref_draws_path).open(f"{model_name}.ref_draws.json"))
        
    model = BayesKitModel(model_path, json.dumps(data))
    return model, ref_draws, posterior_origin


def stan_posterior(model_name, posterior_path):
    try:  # try to load posterior from PDB
        posterior_origin = "pdb"
        path = os.path.join(posterior_path, "posteriordb/posterior_database/")
        
        pdb = PosteriorDatabase(path)
        posterior = pdb.posterior(model_name)
        
        model_path = posterior.model.code_file_path("stan")
        data = posterior.data.values()
        ref_draws = posterior.reference_draws()
        
    except:  # load posterior from custom model
        posterior_origin = "custom"
        path = os.path.join(posterior_path, model_name)
        
        model_path = os.path.join(path, f"{model_name}.stan")
        
        data_path = os.path.join(path, f"{model_name}.data.json")
        data = json.load(open(data_path, "r"))
        
        ref_draws_path = os.path.join(path, f"{model_name}.ref_draws.json.zip")
        ref_draws = json.load(ZipFile(ref_draws_path).open(f"{model_name}.ref_draws.json"))
        
    model = CmdStanModel(stan_file=model_path)
    return model, data, ref_draws, posterior_origin


@lru_cache(maxsize=128)
def get_posterior(model_name, posterior_path, method):
    if method == "stan":
        return stan_posterior(model_name, posterior_path)
    elif method == "bayeskit":
        return bayes_kit_posterior(model_name, posterior_path)
    else:
        raise ValueError(f"Method {method} not implemented")


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