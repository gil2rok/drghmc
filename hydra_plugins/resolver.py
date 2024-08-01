from omegaconf import OmegaConf

from src.utils.save_utils import get_group


def sampler_param_resolver(sampler_params_dict):
    return get_group(sampler_params_dict)

if OmegaConf._get_resolver("concat_sampler_params") is None:
    OmegaConf.register_new_resolver("concat_sampler_params", sampler_param_resolver)