import os
import yaml

from hydra.core.hydra_config import HydraConfig
import numpy as np


def get_history(config):
    history_path = os.path.join(
        HydraConfig.get().runtime.output_dir,
        f"history__chain={config.sampler.chain}.npz",
    )
    return np.load(history_path)


def save_dict_to_npz(dictionary, filename):
    data_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(data_dir, f"{filename}.npz")
    np.savez(path, **dictionary)


def get_group(sampler_params_dict):
    """ Maps from config.sampler.params to a string group name. 
    
    Used in custom Hydra interpolation resolver to format hyper-parameter group for 
    subdir names. To access elsewhere, use `HydraConfig.get().runtime.output_dir`.
    
    Chain num is NOT included in group name intentionally.
    """
    return "__".join(
        sorted([f"{k}={v}" for k, v in sampler_params_dict.items() if v is not None])
    )