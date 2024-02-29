import argparse
import os
import yaml

import submitit
import wandb

# Set API key
os.environ["WANDB_API_KEY"] = "9f41d365fc09eb8ed25cbe56c00b4c8bca11852d"
os.environ["WANDB_DISABLE_SERVICE"] = "True"


def launch_sweep(sweep_config_path):
    # initialize wandb sweep
    with open(sweep_config_path, "r") as file:
        sweep_config = yaml.safe_load(file)
    project = sweep_config["parameters"]["experiment"]["value"]
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    # initialize slurm executor
    executor = submitit.AutoExecutor(folder="logs/slurm_%j")
    executor.update_parameters(
        nodes=1,
        slurm_ntasks_per_node=128,
        slurm_time="1-00:00:00",
    )

    # run wandb sweep on slurm executor
    executor.submit(
        lambda sweep_id: wandb.agent(sweep_id), 
        sweep_id
    )


def main(experiment):
    sampler_configs = os.listdir(os.path.join("experiments", experiment))
    sampler_configs.sort(reverse=True) # run NUTS first

    sampler_configs = ["drghmc_config.yaml"]
    for sampler_config in sampler_configs:
        config_path = os.path.join("experiments", experiment, sampler_config)
        launch_sweep(config_path)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--experiment",
        type=str,
        help="name of experiment to run",
    )
    args = argparse.parse_args()

    main(args.experiment)
