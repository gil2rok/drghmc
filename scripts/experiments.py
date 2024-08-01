import subprocess
import logging

import hydra

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True
logging.basicConfig(level=logging.ERROR)
hydra.utils.log.setLevel(logging.ERROR)


def baseline(posterior):
    subprocess.run(["python", "-m", "scripts.run_nuts", f"posterior.name={posterior}", "wandb.tags=baseline"])
    subprocess.run(["python", "-m", "scripts.run_nuts", f"posterior.name={posterior}", "wandb.tags=baseline", "sampler.params.adapt_metric=False"]) # for model to find appropriate nuts step size
    subprocess.run(["python", "-m", "scripts.run_drhmc", f"posterior.name={posterior}", "wandb.tags=baseline"])
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=baseline"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=sampler_type", "tags=baseline"])


def step_size_factor(posterior):
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=step_size_factor", "sampler.params.step_size_factor=0.5,1,2,4,6,8,10,12"])
    subprocess.run(["python", "-m", "scripts.run_drhmc", f"posterior.name={posterior}", "wandb.tags=step_size_factor", "sampler.params.step_size_factor=0.5,1,2,4,6,8,10,12"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=step_size_factor", "tags=step_size_factor"])


def max_proposals(posterior):
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=max_proposals", "sampler.params.max_proposals=2,3,4,5"])
    subprocess.run(["python", "-m", "scripts.run_drhmc", f"posterior.name={posterior}", "wandb.tags=max_proposals", "sampler.params.max_proposals=2,3,4,5"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=max_proposals", "tags=max_proposals"])


def damping(posterior):
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=damping", "sampler.params.damping=0.01, 0.04, 0.08, 0.12, 0.16, 0.2"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=damping", "tags=damping"])


def reduction_factor(posterior):
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=reduction_factor", "sampler.params.reduction_factor=2,4,6,8,10,12"])
    subprocess.run(["python", "-m", "scripts.run_drhmc", f"posterior.name={posterior}", "wandb.tags=reduction_factor", "sampler.params.reduction_factor=2,4,6,8,10,12"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=reduction_factor", "tags=reduction_factor"])


def probabilistic(posterior):
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=probabilistic", "sampler.params.probabilistic=True,False"])
    subprocess.run(["python", "-m", "scripts.run_drhmc", f"posterior.name={posterior}", "wandb.tags=probabilistic", "sampler.params.probabilistic=True,False"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=probabilistic", "tags=probabilistic"])


def adapt_metric(posterior):
    subprocess.run(["python", "-m", "scripts.run_nuts", f"posterior.name={posterior}", "wandb.tags=adapt_metric", "sampler.params.adapt_metric=True,False"])
    subprocess.run(["python", "-m", "scripts.run_drhmc", f"posterior.name={posterior}", "wandb.tags=adapt_metric", "sampler.params.adapt_metric=True,False"])
    subprocess.run(["python", "-m", "scripts.run_drghmc", f"posterior.name={posterior}", "wandb.tags=adapt_metric", "sampler.params.adapt_metric=True,False"])
    subprocess.run(["python", "-m", "scripts.generate_figures", f"posterior.name={posterior}", "hyper_param=adapt_metric", "tags=adapt_metric"])


def main():
    posteriors = [
        # "funnel10",
        # "eight_schools-eight_schools_centered",
        # "normal50",
        "irt_2pl",
        "stochastic_volatility", 
    ]

    for posterior in posteriors:
        baseline(posterior)    
        # step_size_factor(posterior) 
        # max_proposals(posterior)
        # damping(posterior)
        # reduction_factor(posterior)
        # probabilistic(posterior)
        # adapt_metric(posterior)


if __name__ == '__main__':
    main()