import argparse
import os

import numpy as np
import wandb

from samplers.drghmc import DrGhmcDiag
from utils.posteriors import get_posterior
from utils.configure_samplers import get_init
from utils.summary_stats import squared_error, processed_ref_draws
from utils.nuts_wandb import get_nuts_step_size, get_nuts_step_count


def configure_sampler(config):
    model, ref_draws, posterior_origin = get_posterior(
        config.posterior, config.posterior_dir, "bayeskit"
    )

    if config.step_size_factor:
        nuts_step_size = get_nuts_step_size(wandb.run.entity, wandb.run.project, f"nuts__chain-{config.chain}")
        init_step_size = nuts_step_size * config.step_size_factor
    elif config.step_size:
        init_step_size = config.step_size

    step_sizes = [
        init_step_size * (config.reduction_factor**-k)
        for k in range(config.max_proposals)
    ]

    if config.step_count_factor:
        nuts_step_counts = get_nuts_step_count(wandb.run.entity, wandb.run.project, f"nuts__chain-{config.chain}")
        init_step_count = int(np.percentile(nuts_step_counts, config.step_count_factor * 100))
    elif config.step_count:
        init_step_count = config.step_count
        
    traj_len = init_step_count * init_step_size
    step_counts = [
        int(traj_len / step_sizes[k]) for k in range(config.max_proposals)
    ]

    init = get_init(ref_draws, config.chain, "bk", posterior_origin)

    return DrGhmcDiag(
        model=model,
        max_proposals=config.max_proposals,
        leapfrog_step_sizes=step_sizes,
        leapfrog_step_counts=step_counts,
        damping=config.dampening,
        # metric_diag=metric,
        init=init,
        seed=config.seed,
        prob_retry=config.probabilistic,
    )


def main(config):
    # configure sampler
    sampler = configure_sampler(config)

    num_params = sampler._model.dims()
    ref_draws = processed_ref_draws(config.posterior_dir, config.posterior)
    ref_draws_squared = np.square(ref_draws)

    # define wandb metrics
    wandb.define_metric("grad_evals", summary="last")
    wandb.define_metric("se1_grad", summary="max")
    wandb.define_metric("se2_grad", summary="max")
    for i in range(num_params):
        wandb.define_metric(f"param_{i}/se1", summary="last", step_metric="grad_evals")
        wandb.define_metric(f"param_{i}/se2", summary="last", step_metric="grad_evals")

    while sampler._model.log_density_gradient.calls < config.grad_evals:
        # generate draws and compute squared error
        draw = sampler.sample()[0]
        se1 = squared_error(draw, ref_draws)
        se2 = squared_error(np.square(draw), ref_draws_squared)

        # log metrics
        
        # for i in range(num_params):
        #     wandb.log(
        #         {
        #             "grad_evals": sampler._model.log_density_gradient.calls,
        #             f"param_{i}/se1": se1[i],
        #             f"param_{i}/se2": se2[i],
        #             f"param_{i}/draws": draw[i],
        #         }
        #     )

    # log squared error / gradient evaluations after all draws are generated
    total_grad_evals = sampler._model.log_density_gradient.calls
    for i in range(num_params):
        wandb.log(
            {
                f"param_{i}/se1_grad": se1[i] / total_grad_evals,
                f"param_{i}/se2_grad": se2[i] / total_grad_evals,
                "se1_grad": se1[i] / total_grad_evals,
                "se2_grad": se2[i] / total_grad_evals,
            }
        )
    
    # log trajectory length of uturns
    for uturn in sampler._uturn_list:
        wandb.log({"uturns/uturns": uturn})
        
    # table = wandb.Table(columns=["trajectory_length"], data=np.array(sampler._uturn_list))
    # wandb.log({"uturns/trajectory_length": table})
    
    # hist = np.histogram(np.array(sampler._uturn_list), density=True)
    # wandb.run.summary.update({"uturns/hist": wandb.Histogram(np_histogram=hist)})
    
    # for uturn in sampler._uturn_list:
    #     wandb.log({"uturns/uturns": uturn})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--project",
        type=str,
        help="name of project to log to",
    )
    parser.add_argument(
        "--posterior",
        type=str,
        help="name of posterior to sample from",
    )
    parser.add_argument(
        "--posterior_dir",
        type=str,
        help="directory to posterior",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="global seed for experiment",
    )
    parser.add_argument(
        "--sampler_type",
        type=str,
        help="type of sampling algorithm to use",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        help="number of burn-in iterations",
    )
    parser.add_argument(
        "--grad_evals",
        type=int,
        help="number of gradient evaluations to run sampling algo for",
    )
    parser.add_argument(
        "--chain",
        type=int,
        help="chain number",
    )
    

    step_count_group = parser.add_mutually_exclusive_group(required=True)
    step_count_group.add_argument(
        "--step_count",
        type=float,
        help="number of leapfrog steps for first proposal",
    )
    step_count_group.add_argument(
        "--step_count_factor",
        type=float,
        help="leapfrog step count for first proposal, computed as this percentile of NUTS step count histograms",
    )

    step_size_group = parser.add_mutually_exclusive_group(required=True)
    step_size_group.add_argument(
        "--step_size",
        type=float,
        help="leapfrog step size for first proposal",
    )
    step_size_group.add_argument(
        "--step_size_factor",
        type=float,
        help="leapfrog step size for first proposal, computed as the NUTS step size multiplied by this factor",
    )

    parser.add_argument(
        "--max_proposals",
        type=int,
        help="maximum number of proposals to make",
    )
    parser.add_argument(
        "--reduction_factor",
        type=float,
        help="factor by which to reduce the step size in subsequent proposals",
    )
    parser.add_argument(
        "--damping",
        type=float,
        help="damping parameter for momentum refresh in generalized HMC",
    )
    parser.add_argument(
        "--probabilistic",
        help="whether to use probabilistic retry",
        action="store_true",
    )

    args = parser.parse_args()
    assert(args.probabilistic == False) # no probabilistic retry

    irrelevant_hyperparams = [
        "project",
        "posterior",
        "posterior_dir",
        "seed",
        "sampler_type",
        "burn_in",
        "grad_evals",
        "damping"
        "probabilistic",
    ]

    group = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and k not in ["chain"] and v is not None
        ]
    )
    run_name = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and v is not None
        ]
    )
    WANDB_RUN_GROUP = group  # set environment var
    wandb.init(
        config=args,
        name=run_name,
        group=group,
        project=args.project,
        # job_type=args.sampler_type,
        save_code=False,
    )
    main(wandb.config)
