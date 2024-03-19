from functools import lru_cache
import os

import numpy as np


@lru_cache(maxsize=20)
def _get_nuts_history(sampler, posterior):
    metric_name = "identity" if sampler.params.metric == 1 else "diag_cov"
    dir_name = f"adapt_metric={sampler.params.adapt_metric}__metric={metric_name}__sampler_type=nuts" 
    fname = f"history__chain={sampler.chain}.npz"
    path = os.path.join("data", f"{posterior.name}", "nuts-baseline", dir_name, fname)

    try:
        history = np.load(path)
    except Exception as e:
        raise FileNotFoundError(
            f"History file not found at\n\t{path}\nwith error:\n\t{e}"
        )

    return history


def get_nuts_step_size(sampler, posterior):
    history = _get_nuts_history(sampler, posterior)
    # all nuts step sizes are the same
    return history["step_size"][0]


def get_nuts_step_counts(sampler, posterior):
    history = _get_nuts_history(sampler, posterior)
    return history["step_count"]


def get_nuts_metric(sampler, posterior):
    pass