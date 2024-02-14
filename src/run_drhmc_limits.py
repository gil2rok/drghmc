import argparse

import numpy as np
import wandb

from utils.posteriors import get_posterior
from samplers.drghmc_limits import DrGhmcDiag


def configure_sampler(config, scale):
    model, ref_draws, posterior_origin = get_posterior(
        config.posterior, config.posterior_dir, "bayeskit"
    )

    seed = int(str(config.global_seed) + str(config.scale_repetitions))
    rng = np.random.default_rng(seed)

    traj_len = config.step_size * config.step_count
    step_sizes = [
        float(config.step_size) * (config.reduction_factor**-k)
        for k in range(config.num_proposals)
    ]
    step_counts = [
        int(traj_len / step_sizes[k]) for k in range(config.num_proposals)
    ]  # const traj len

    non_scale_params = rng.normal(loc=0.0, scale=np.exp(scale), size=model.dimensions - 1)
    init = np.concatenate((np.array([scale]), non_scale_params))

    sampler = DrGhmcDiag(
        model=model,
        max_proposals=config.num_proposals,
        leapfrog_step_sizes=step_sizes,
        leapfrog_step_counts=step_counts,
        damping=1.0,
        metric_diag=None,
        init=init,
        seed=seed,
        prob_retry=config.probabilistic,
    )
    return sampler


def main(config):
    
    wandb.define_metric("scale")
    wandb.define_metric("a1", step_metric="scale")
    wandb.define_metric("a2", step_metric="scale")
    wandb.define_metric("a3", step_metric="scale")
    wandb.define_metric("a4", step_metric="scale")
    
    scale_params = np.linspace(
        config.scale_min, config.scale_max, config.scale_linspace
    )
    
    for scale in scale_params:
        sampler = configure_sampler(config, scale)

        while sampler._model.log_density_gradient.calls < config.grad_evals:
            try: 
                sampler.sample()

                wandb.log(
                    {
                        "scale": scale,
                        "a1": np.exp(sampler._accept1),
                        "a2": np.exp(sampler._accept2),
                        "a3": np.exp(sampler._accept3),
                        "a4": np.exp(sampler._accept4),
                    }
                )
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--global_seed",
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
        "--step_size",
        type=float,
        help="leapfrog step size for first proposal",
    )
    parser.add_argument(
        "--step_count",
        type=int,
        help="number of leapfrog steps for first proposal",
    )
    parser.add_argument(
        "--num_proposals",
        type=int,
        help="number of delayed rejection proposals to make",
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        help="reduction factor for decreasing step size in every subsequent proposal",
    )
    parser.add_argument(
        "--probabilistic",
        type=bool,
        help="whether to probabilistically attempt another proposal",
    )
    parser.add_argument(
        "--scale_min",
        type=float,
        help="minimum scale parameter",
    )
    parser.add_argument(
        "--scale_max",
        type=float,
        help="maximum scale parameter",
    )
    parser.add_argument(
        "--scale_linspace",
        type=int,
        help="divide [scale_min, scale_max] into scale_linspace even intervals",
    )
    parser.add_argument(
        "--scale_repetitions",
        type=float,
        help="number of times to repeat each scale parameter",
    )

    args = parser.parse_args()

    irrelevant_hyperparams = [
        "posterior",
        "posterior_dir",
        "global_seed",
        "sampler_type",
        "burn_in",
        "grad_evals",
        "scale_min",
        "scale_max",
        "scale_linspace",
    ]

    group = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and k not in ["scale_repetitions"]
        ]
    )
    run_name = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams
        ]
    )

    WANDB_RUN_GROUP = group  # set environment var
    wandb.init(
        config=args,
        # name=run_name,
        group=group,
        # job_type=args.sampler_type,
        save_code=False,
    )
    main(wandb.config)
