import os
import yaml

import numpy as np
import polars as pl


def get_data_dir(config):
    irrelevant_keys = set(
        [
            "experiment",
            "posterior",
            "posterior_dir",
            "group",
            "generate_history",
            "generate_metrics",
        ]
    )

    parent_dir = "__".join(
        sorted([f"{k}={v}" for k, v in config.items() if k not in irrelevant_keys])
    )
    data_dir = os.path.join("data", config.posterior, parent_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def save_to_npz(dictionary, config, filename):
    data_dir = get_data_dir(config)
    path = os.path.join(data_dir, f"{filename}.npz")
    np.savez(path, **dictionary)


def save_to_parquet(df, config, filename):
    data_dir = get_data_dir(config)
    path = os.path.join(data_dir, f"{filename}.parquet")
    df.write_parquet(path)


def save_fingerprint(config):
    data_dir = get_data_dir(config)
    path = os.path.join(data_dir, "fingerprint.yaml")

    fingerprint = {k: v for k, v in config.items()}

    with open(path, "w") as file:
        yaml.dump(fingerprint, file, default_flow_style=False)


def save_fingerprint2(config):
    data_dir = get_data_dir(config)
    path = os.path.join(data_dir, "fingerprint.yaml")
    with open(path, "w") as file:
        for k, v in config.items():
            if k == "generate_history":
                v = "True"
            file.write(f"{k}: {v}\n")
