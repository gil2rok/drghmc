import argparse
import os
import subprocess
import yaml

import wandb

# Set API key
os.environ["WANDB_API_KEY"] = "9f41d365fc09eb8ed25cbe56c00b4c8bca11852d"

# Gather nodes allocated to current slurm job
result = subprocess.run(["scontrol", "show", "hostnames"], stdout=subprocess.PIPE)
node_list = result.stdout.decode("utf-8").split("\n")[:-1]


def launch_sweep(config_yaml, project_name):
    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(config_dict, project=project_name)

    sp = []
    for node in node_list:
        sp.append(
            subprocess.Popen(
                [
                    "srun",
                    "--nodes=1",
                    "--ntasks=1",
                    "-w",
                    node,
                    "src/runners/start_agent.sh",
                    sweep_id,
                    project_name,
                ]
            )
        )
    exit_codes = [p.wait() for p in sp]  # wait for processes to finish
    return exit_codes


def main(experiment):
    project_name = f"dr-{experiment}"
    wandb.init(project=project_name)

    drghmc_config_path = os.path.join(
        "experiments", experiment, "configs", "drghmc_sweep.yaml"
    )
    launch_sweep(drghmc_config_path, project_name)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--experiment",
        type=str,
        help="name of experiment to run",
    )

    args = argparse.parse_args()
    main(args.experiment)
