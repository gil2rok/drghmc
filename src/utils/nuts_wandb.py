from functools import lru_cache
import json
import os

import numpy as np
import wandb

@lru_cache(maxsize=20)
def get_nuts_run(entity, project, nuts_run_name):
    try:
        nuts_run_id = (
            wandb.Api()
            .runs(
                path=os.path.join(entity, project),
                filters={"display_name": nuts_run_name},
            )
            .next()
            .id
        ) # default behavior returns newest run with name nuts_run_name
    except:
        raise ValueError(
            f"Could not find run with name {nuts_run_name} in project {project} for entity {entity}"
        )

    nuts_run = wandb.Api().run(os.path.join(entity, project, nuts_run_id))

    try:
        assert nuts_run.state == "finished"
    except:
        raise ValueError(
            f"Run {nuts_run_name} in project {project} for entity {entity} is not finished"
        )

    return nuts_run

@lru_cache(maxsize=10)
def get_nuts_step_size(entity, project, nuts_run_name):
    nuts_run = get_nuts_run(entity, project, nuts_run_name)
    return nuts_run.summary.get("nuts/step_size")

@lru_cache(maxsize=10)
def get_nuts_metric(entity, project, nuts_run_name):
    nuts_run = get_nuts_run(entity, project, nuts_run_name)   

    metric_path= nuts_run.summary["nuts/metric"]["path"]
    metric_file = nuts_run.file(metric_path).download(exist_ok=True)
    metric_json = json.load(metric_file)
    metric = np.array(metric_json["data"]).squeeze()
    
    return metric

@lru_cache(maxsize=10)
def get_nuts_step_count(entity, project, nuts_run_name):
    nuts_run = get_nuts_run(entity, project, nuts_run_name)

    step_counts_path= nuts_run.summary["nuts/step_counts"]["path"]
    step_counts_file = nuts_run.file(step_counts_path).download(exist_ok=True)
    step_counts_json = json.load(step_counts_file)
    step_counts = np.array(step_counts_json["data"], dtype=int).squeeze()

    return step_counts