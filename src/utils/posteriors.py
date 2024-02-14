import json, os, zipfile
from functools import lru_cache

import bridgestan as bs
from cmdstanpy import CmdStanModel
import numpy as np
from posteriordb import PosteriorDatabase

from .typing import GradModel


class BayesKitModel(GradModel):
    def __init__(self, model_path, data):
        self.bsmodel = bs.StanModel.from_stan_file(
            stan_file=model_path,
            model_data=data,
            seed=1234,
            make_args=["TBB_CXX_TYPE=gcc"]
        )
        self.dimensions = self.bsmodel.param_unc_num()
        
    def log_density(self, x):
        return self.bsmodel.log_density(x)
    
    def log_density_gradient(self, x):
        log_density, gradient = self.bsmodel.log_density_gradient(x)
        
        if np.isnan(log_density) or np.isnan(gradient).any():
            raise ValueError("NaN values in log density or gradient")
        
        return (log_density, gradient)
    
    def unconstrain(self, x):
        return self.bsmodel.param_unconstrain(x)
    
    def constrain(self, x):
        return self.bsmodel.param_constrain(x)
    
    def dims(self):
        return self.dimensions


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
        ref_draws = json.load(zipfile.ZipFile(ref_draws_path).open(f"{model_name}.ref_draws.json"))
        
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
        ref_draws = json.load(zipfile.ZipFile(ref_draws_path).open(f"{model_name}.ref_draws.json"))
        
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