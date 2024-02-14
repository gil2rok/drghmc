import argparse
from collections import defaultdict
import logging
import math
import os

import numpy as np
import wandb

from samplers.drghmc import DrGhmcDiag
from utils.posteriors import get_posterior
from utils.configure_samplers import get_init
from utils.summary_stats import squared_error, processed_ref_draws, streaming_se, streaming_relative_se, streaming_avg, relative_squared_error2, squared_error2
from utils.nuts_wandb import get_nuts_step_size, get_nuts_metric


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

    if config.step_count_method == "const_step_count":
        # const number of steps (ghmc)
        step_counts = [1 for _ in range(config.max_proposals)]
    elif config.step_count_method == "const_traj_len":
        # const trajectory length (drhmc)
        init_step_count = 1
        traj_len = init_step_count * init_step_size
        step_counts = [
            int(traj_len / step_sizes[k]) for k in range(config.max_proposals)
        ]

    if config.metric == 0:
        metric = get_nuts_metric(wandb.run.entity, wandb.run.project, f"nuts__chain-{config.chain}")
        print(type(metric), metric)
    elif config.metric == 1:
        metric = None

    init = get_init(ref_draws, config.chain, "bk", posterior_origin)
    
    damping = float(config.damping) # w&b casts damping float 1.0 to 1

    return DrGhmcDiag(
        model=model,
        max_proposals=config.max_proposals,
        leapfrog_step_sizes=step_sizes,
        leapfrog_step_counts=step_counts,
        damping=damping,
        metric_diag=metric,
        init=init,
        seed=config.seed,
        prob_retry=config.probabilistic,
    )

def main(config):
    # configure sampler
    sampler = configure_sampler(config)
    num_params = sampler._model.dims()
    
    # get reference draws
    ref_draws = processed_ref_draws(config.posterior_dir, config.posterior)
    ref_draws_squared = np.square(ref_draws)
    avg_ref_draws = np.mean(ref_draws, axis=(0,2)) # [num_params]
    avg_ref_draws_squared = np.mean(ref_draws_squared, axis=(0,2)) # [num_params]

    # define wandb metrics
    wandb.define_metric("grad_evals", summary="max")
    wandb.define_metric("max_se1", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_se2", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_rse1", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_rse2", step_metric="grad_evals", summary="last")
    
    for i in range(num_params):
        wandb.define_metric(f"param_{i}/se1_grad", summary="last")
        wandb.define_metric(f"param_{i}/se2_grad", summary="last")
        wandb.define_metric(f"param_{i}/se1", summary="last")
        wandb.define_metric(f"param_{i}/se2", summary="last")
        wandb.define_metric(f"param_{i}/rse1_grad", summary="last")
        wandb.define_metric(f"param_{i}/rse2_grad", summary="last")
        wandb.define_metric(f"param_{i}/rse1", summary="last")
        wandb.define_metric(f"param_{i}/rse2", summary="last")

    # init lists/params
    jump_dist, acceptances, rejections, uturns = [], [], [], []
    uturn_dict = defaultdict(int)   
    prev_draw, samples_since_rejection = None, 0
    avg_draws = avg_draws_squared = counter = 0

    log_freq = max(1, config.grad_evals // 1000)
    prev_log = 0

    # run sampler and log metrics
    while sampler._model.log_density_gradient.calls < config.grad_evals:
        # generate draws
        draw = sampler.sample()[0]
        
        # update uturns
        for old_draw, old_dist in tuple(uturn_dict.items()):
            dist = np.linalg.norm(draw - old_draw) # squared distance 
            if dist < old_dist:
                uturns.append([dist])
                del uturn_dict[old_draw]
            elif dist >= old_dist:
                uturn_dict[old_draw] = dist
        uturn_dict[tuple(draw)] = 0.0
        
        # compute squared error and relative squared error
        avg_draws = streaming_avg(draw, avg_draws, counter)
        avg_draws_squared = streaming_avg(np.square(draw), avg_draws_squared, counter)
        
        se1 = squared_error2(avg_draws, avg_ref_draws)
        se2 = squared_error2(avg_draws_squared, avg_ref_draws_squared)
        
        rse1 = relative_squared_error2(avg_draws, avg_ref_draws)
        rse2 = relative_squared_error2(avg_draws_squared, avg_ref_draws_squared)
        
        counter += 1

        # log squared error
        if sampler._model.log_density_gradient.calls - prev_log >= log_freq or prev_log == 0:
            wandb.log(
                {
                    "grad_evals": sampler._model.log_density_gradient.calls,
                    "max_se1": max(se1),
                    "max_se2": max(se2),
                    "max_rse1": max(rse1),
                    "max_rse2": max(rse2),
                }
            )
            prev_log = sampler._model.log_density_gradient.calls

        # compute acceptances, jump_distance, and sequential acceptances
        acceptances.append([sampler._acceptance_list[-1]])
        if prev_draw is not None:
            jump_dist.append([np.linalg.norm(draw - prev_draw) ** 2])
        prev_draw = draw
        if sampler._acceptance_list[-1] >= 1:
            samples_since_rejection += 1
        else:
            rejections.append([samples_since_rejection])
            samples_since_rejection = 0

    # for each model parameter, log squared error and squared error per gradient evaluation
    total_grad_evals = sampler._model.log_density_gradient.calls
    for i in range(num_params):
        wandb.log(
            {
                # log squared error and squared error/gradient for each model parameter
                f"param_{i}/se1_grad": se1[i] / total_grad_evals,
                f"param_{i}/se2_grad": se2[i] / total_grad_evals,
                f"param_{i}/se1": se1[i],
                f"param_{i}/se2": se2[i],
                f"param_{i}/rse1_grad": rse1[i] / total_grad_evals,
                f"param_{i}/rse2_grad": rse2[i] / total_grad_evals,
                f"param_{i}/rse1": rse1[i],
                f"param_{i}/rse2": rse2[i],
            }
        )

    wandb.log(
        {
            # log final squared error
            "grad_evals": sampler._model.log_density_gradient.calls,
            "max_se1": max(se1),
            "max_se2": max(se2),
            "max_rse1": max(rse1),
            "max_rse2": max(rse2),
            # log squared error and squared error per gradient evaluation as scalars
            "squared_error/se1": max(se1),
            "squared_error/se2": max(se2),
            "squared_error/se1_grad": max(se1) / total_grad_evals,
            "squared_error/se2_grad": max(se2) / total_grad_evals,
            "squared_error/rse1": max(rse1),
            "squared_error/rse2": max(rse2),
            "squared_error/rse1_grad": max(rse1) / total_grad_evals,
            "squared_error/rse2_grad": max(rse2) / total_grad_evals,
            # log diagnostics
            "diagnostics/proposal": sampler._proposal_nans,
            "diagnostics/ghost": sampler._ghost_nans,
            "diagnostics/nans": sampler._proposal_nans + sampler._ghost_nans,
            "diagnostics/num_draws": len(acceptances),
            "diagnostics/nans_fraction": (sampler._proposal_nans + sampler._ghost_nans) / len(acceptances),
        }
    )
    
    # downsample and create wandb tables
    table_dict = dict()
    if acceptances:
        acceptances = acceptances[::math.ceil(len(acceptances) / 10000)]
        acceptances_table = wandb.Table(columns=["acceptances"], data=acceptances)
        table_dict["acceptances/acceptances"] = acceptances_table
    if jump_dist:
        jump_dist = jump_dist[::math.ceil(len(jump_dist) / 10000)]
        jump_table = wandb.Table(columns=["jump_dist"], data=jump_dist)
        table_dict["jump_dist/jump_dist"] = jump_table
    if rejections:
        rejections = rejections[::math.ceil(len(rejections) / 10000)]
        rejection_table = wandb.Table(columns=["rejections"], data=rejections)
        table_dict["acceptances/rejection_frequency"] = rejection_table
    if uturns:
        uturns = uturns[::math.ceil(len(uturns) / 10000)]
        uturns_table = wandb.Table(columns=["uturns"], data=uturns)
        table_dict["uturns/uturns"] = uturns_table
    wandb.log(table_dict)


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
    parser.add_argument(
        "--step_count_method",
        type=str,
        help="method to use for leapfrog step counts in subsequent proposals. As a generalized HMC method, the first step count is always one. Can specificy a constant trajectory length that increases the step count in subsequent proposals, or a constant step count that uses the same step count of one for all proposals",
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
        "--metric",
        type=int,
        help="metric type to use for preconditioning",
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
        "metric",
        "probabilistic",
    ]

    group_name = "__".join(
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

    WANDB_RUN_GROUP = group_name  # set environment var
    wandb.init(
        config=args,
        #name=run_name,
        group=group_name,
        project=args.project,
        job_type=args.sampler_type,
        save_code=False,
    )
    main(wandb.config)
