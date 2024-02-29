import os
import yaml

import numpy as np
import polars as pl
import tqdm
import wandb

from utils.argument_parsers import summarize_argument_parser
from utils.save_utils import save_to_parquet


def fingerprint_summary(row_dict, root, dir):
    try:
        path = os.path.join(root, dir, "fingerprint.yaml")
        with open(path, "r") as file:
            fingerprint = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fingerprint file not found at\n\t{path}")
    
    # ignore irrelevant fingerprint keys from _base_argument_parser()
    irrelevant_keys = [
        "experiment",
        "posterior",
        "posterior_dir",
        "seed",
        "generate_history",
        "generate_metrics",
        "burn_in",
        "gradient_budget",
    ]
    
    for k, v in fingerprint.items():
        if k not in irrelevant_keys:
            row_dict[k] = v
    
    # create hyperparameter string for easy grouping. ignore irrelevant keys and chain
    hyperparameter_str = "__".join(
        [f"{k}={v}" for k, v in fingerprint.items() if k not in irrelevant_keys and k not in ["chain"]]
    )
    row_dict["hparams"] = hyperparameter_str
    
    return row_dict


def history_summary(row_dict, root, dir):
    try:
        path = os.path.join(root, dir, "history.npz")
        history = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"History file not found at\n\t{path}")
    
    row_dict["num_nans"] = history["num_nans"].sum()
    
    return row_dict


def metrics_summary(row_dict, root, dir):
    try:
        path = os.path.join(root, dir, "metrics.npz")
        metrics = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics file not found at\n\t{path}")
    
    row_dict["max_se1"] = metrics["max_se1"][-1]
    row_dict["max_se2"] = metrics["max_se2"][-1]
    
    try:
        row_dict["max_rse1"] = metrics["max_rse1"][-1]
        row_dict["max_rse2"] = metrics["max_rse2"][-1]
    except:
        pass
    
    for idx, param_se1 in enumerate(metrics["se1"][-1]):
        row_dict[f"p{idx}_se1"] = param_se1
        
    for idx, param_se2 in enumerate(metrics["se2"][-1]):
        row_dict[f"p{idx}_se2"] = param_se2
        
    try:
        for idx, param_rse1 in enumerate(metrics["rse1"][-1]):
            row_dict[f"p{idx}_rse1"] = param_rse1
            
        for idx, param_rse2 in enumerate(metrics["rse2"][-1]):
            row_dict[f"p{idx}_rse2"] = param_rse2
    except:
        pass

    return row_dict


def get_schema(row_dict):
    schema = {
        "hparams": pl.String,
        "num_nans": pl.UInt64,
        
        "max_se1": pl.Float32,
        "max_se2": pl.Float32,
        "max_rse1": pl.Float32,
        "max_rse2": pl.Float32,
    }

    num_params = len([k for k in row_dict.keys() if "p" in k and "se1" in k])
    for i in range(num_params):
        schema[f"p{i}_se1"] = pl.Float32
        schema[f"p{i}_se2"] = pl.Float32
        schema[f"p{i}_rse1"] = pl.Float32
        schema[f"p{i}_rse2"] = pl.Float32
    
    return schema


def main(config):
    rows = []
    
    path = os.path.join("data", config.experiment)
    root, dirs, _ = next(os.walk(path))
    
    for dir in tqdm.tqdm(dirs, desc="Summarizing"):
        row_dict = dict()
        
        row_dict = fingerprint_summary(row_dict, root, dir)
        row_dict = history_summary(row_dict, root, dir)
        row_dict = metrics_summary(row_dict, root, dir)
        rows.append(row_dict)

    schema = None # get_schema(row_dict)
    df = pl.DataFrame(data=rows, schema=schema)
    save_to_parquet(df, config, "summary")


if __name__ == "__main__":
    parser = summarize_argument_parser()
    args = parser.parse_args()

    group_name = job_type = "summarize"
    WANDB_RUN_GROUP = group_name
    wandb.init(
        config=args,
        group=group_name,
        project=args.experiment,
        job_type=job_type,
    )
    main(wandb.config)
