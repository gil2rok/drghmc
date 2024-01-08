import argparse
import os
import yaml

import submitit
import wandb

# Set API key
os.environ["WANDB_API_KEY"] = "9f41d365fc09eb8ed25cbe56c00b4c8bca11852d"

def launch_job(sweep_id):
    wandb.login(key="9f41d365fc09eb8ed25cbe56c00b4c8bca11852d", relogin=True)
    wandb.agent(sweep_id)

def launch_sweep(sweep_config_path):
    # initialize wandb sweep
    with open(sweep_config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep=sweep_config)

    # initialize slurm executor
    executor = submitit.SlurmExecutor(folder="logs")
    executor.update_parameters(nodes=1, ntasks_per_node=1)

    # run wandb sweep on slurm executor
    executor.submit(
        launch_job,
        sweep_id,
    )

    # function = submitit.helpers.CommandFunction(
    #     command=["wandb", "agent", f"dr-funnel10/gilad-turok/{sweep_id}"],
    #     env={"WANDB_API_KEY": "9f41d365fc09eb8ed25cbe56c00b4c8bca11852d"}
    # )
    # executor.submit(function)

def main(experiment):
    wandb.init(project=f"dr-{experiment}")
    drghmc_config_path = os.path.join("experiments", experiment, "configs", "drghmc_sweep.yaml")
    launch_sweep(drghmc_config_path)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--experiment",
        type=str,
        help="name of experiment to run",
    )

    args = argparse.parse_args()
    main(args.experiment)