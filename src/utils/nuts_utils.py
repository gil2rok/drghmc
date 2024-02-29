from functools import lru_cache
import os

import numpy as np


@lru_cache(maxsize=20)
def _get_nuts_history(config):
    dir_name = "__".join(
        sorted(
            [
                f"burn_in={config.burn_in}",
                f"chain={config.chain}",
                f"gradient_budget={config.gradient_budget}",
                f"metric=identity",
                "sampler_type=nuts",
                f"seed={config.seed}",
            ]
        )
    )
    path = os.path.join("data", f"{config.experiment}", dir_name, "history.npz")

    try:
        history = np.load(path)
    except Exception as e:
        raise FileNotFoundError(
            f"History file not found at\n\t{path}\nwith error:\n\t{e}"
        )

    return history


def get_nuts_step_size(config):
    history = _get_nuts_history(config)
    # all nuts step sizes are the same
    return history["step_size"][0]


def get_nuts_step_counts(config):
    history = _get_nuts_history(config)
    return history["step_count"]
