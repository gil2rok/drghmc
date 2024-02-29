import functools

import bridgestan as bs
import numpy as np

from .typing import GradModel


def grad_counter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


class BayesKitModel(GradModel):
    def __init__(self, model_path, data_path):
        self.bsmodel = bs.StanModel(
            model_lib=model_path,
            data=data_path,
            seed=1234,
            make_args=["STAN_THREADS=True", "TBB_CXX_TYPE=gcc"]
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
