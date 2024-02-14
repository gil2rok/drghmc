import argparse
from collections import namedtuple, defaultdict
import logging
import os
import math

import numpy as np
import wandb

from utils.configure_samplers import stan_nuts
from utils.summary_stats import squared_error, processed_ref_draws, relative_squared_error, streaming_avg, squared_error2, relative_squared_error2

logging.basicConfig(level=logging.INFO, filename=os.path.join("logs", "run_nuts.log"), filemode="w")


HyperParamsTuple = namedtuple(
    "hyper_params",
    [
        "posterior",
        "burn_in",
        "grad_evals",
        "global_seed",
        "chain",
        "save_dir",
        "posterior_dir",
        "bridgestan_dir",
    ],
)


def grad_eval_filter(nuts, config):
    nuts_np = nuts.draws(concat_chains=True)
    grad_evals = np.cumsum(nuts_np[:, 4])  # [num_draws]
    idx = np.argwhere(grad_evals > config.grad_evals)[0][0]
    return nuts_np[:idx, :]


def process_nuts(nuts, config):
    nuts_np = grad_eval_filter(nuts, config)
    grad_evals = np.cumsum(nuts_np[:, 4])  # [num_draws]
    
    draws = nuts_np[:, 7:]  # [num_draws, num_params]
    num_draws, num_params = draws.shape

    ref_draws = processed_ref_draws(config.posterior_dir, config.posterior)
    ref_draws_squared = np.square(ref_draws)
    
    avg_ref_draws = np.mean(ref_draws, axis=(0, 2))
    avg_ref_draws_squared = np.mean(ref_draws_squared, axis=(0, 2))
    
    se1, se2, rse1, rse2 = [], [], [], []
    avg_draws = avg_draws_squared = 0
    for idx in range(num_draws):
        avg_draws = streaming_avg(draws[idx, :], avg_draws, idx)
        avg_draws_squared = streaming_avg(np.square(draws[idx, :]), avg_draws_squared, idx)
        
        se1.append(squared_error2(avg_draws, avg_ref_draws))
        se2.append(squared_error2(avg_draws_squared, avg_ref_draws_squared))
        
        rse1.append(relative_squared_error2(avg_draws, avg_ref_draws))
        rse2.append(relative_squared_error2(avg_draws_squared, avg_ref_draws_squared))

    # extract step_size
    step_sizes = nuts_np[:, 2]

    # extract step_counts
    step_counts = nuts_np[:, 4]

    metric = nuts.metric.squeeze()

    # extract squared jump distance
    jump_dist = np.linalg.norm(draws[:-1, :] - draws[1:, :], axis=1)
    jump_dist = [[jump] for jump in jump_dist] # format for wandb table

    # u-turns
    uturns, uturn_dict = [], defaultdict(int)
    for draw in draws:
        for old_draw, old_dist in tuple(uturn_dict.items()):
            dist = np.linalg.norm(draw - old_draw) # squared distance 
            if dist < old_dist:
                uturns.append([dist])
                del uturn_dict[old_draw]
            elif dist >= old_dist:
                uturn_dict[old_draw] = dist
        uturn_dict[tuple(draw)] = 0.0

    return draws, grad_evals, se1, se2, rse1, rse2, step_sizes, step_counts, metric, jump_dist, uturns

def create_table(draws, grad_evals, se1, se2):
    # add draws to wandb table
    num_draws, num_params = draws.shape

    param_names = [f"param_{i}" for i in range(num_params)]
    se1_names = [f"se1_param_{i}" for i in range(num_params)]
    se2_names = [f"se2_param_{i}" for i in range(num_params)]

    col_names = param_names + se1_names + se2_names
    data = np.concatenate((draws, se1, se2), axis=1)
    table = wandb.Table(data=data, columns=col_names)

    # add grad_evals to wandb table
    table.add_column(name="grad_evals", data=grad_evals)

    return table


def configure_sampler(config):
    hp = HyperParamsTuple(
        posterior=config.posterior,
        burn_in=None,
        grad_evals=None,
        global_seed=config.seed,
        chain=config.chain,
        save_dir=None,
        posterior_dir="posteriors/",
        bridgestan_dir="none",
    )

    model, data, seed, init, inv_metric = stan_nuts(hp)
    return model, data, seed, init, inv_metric


def generate_samples(model, data, seed, init, inv_metric):
    nuts = model.sample(
        data=data,
        chains=1,
        seed=seed,
        inits=init,
        iter_sampling=int(wandb.config.grad_evals / 5),
        metric=inv_metric,
        adapt_init_phase=0,  # b/c init from reference draw
        show_console=False,
    )
    return nuts


def log(draws, grad_evals, se1, se2):
    table = create_table(draws, grad_evals, se1, se2)

    num_draws, num_params = draws.shape
    param_names = [f"param_{i}" for i in range(num_params)]
    se1_names = [f"se1_param_{i}" for i in range(num_params)]
    se2_names = [f"se2_param_{i}" for i in range(num_params)]
    sampler = WANDB_RUN_GROUP

    # create and log charts for each model param
    for idx in range(num_params):
        # create param histogram
        param_hist = wandb.plot.histogram(
            table,
            param_names[idx],
            title=sampler,
        )

        # create param trace
        param_trace = wandb.plot.line(
            table,
            x="grad_evals",
            y=param_names[idx],
            title=sampler,
        )

        # create se1 lineplot
        se1_lineplot = wandb.plot.line(
            table,
            x="grad_evals",
            y=se1_names[idx],
            title=sampler,
        )

        # create se2 lineplot
        se2_lineplot = wandb.plot.line(
            table,
            x="grad_evals",
            y=se2_names[idx],
            title=sampler,
        )

        # log param histogram
        wandb.log(
            {
                f"param_{idx}_hist/{sampler}": param_hist,
                f"param_{idx}_trace/{sampler}": param_trace,
                f"param_{idx}_se1/{sampler}": se1_lineplot,
                f"param_{idx}_se2/{sampler}": se2_lineplot,
            }
        )

    wandb.log(
        {
            wandb.summary["run_name"]: table,
        }
    )


def log2(draws, grad_evals, se1, se2):
    num_draws, num_params = draws.shape

    # custom x-axis called param
    # # wandb.define_metric("param")
    # wandb.define_metric("se1/grad", step_metric="param", summary="max")
    # wandb.define_metric("se2/grad", step_metric="param", summary="max")

    for idx in range(num_params):
        # wandb.log({
        #     "se1_grad": se1[-1, idx] / grad_evals[-1],
        #     "se2_grad": se2[-1, idx] / grad_evals[-1],
        #     "param": idx,
        # })

        # create wandb table
        data = np.concatenate(
            (
                draws[:, idx : idx + 1],
                se1[:, idx : idx + 1],
                se2[:, idx : idx + 1],
                grad_evals.reshape(-1, 1),
            ),
            axis=1,
        )
        table = wandb.Table(data=data, columns=["draws", "se1", "se2", "grad_evals"])

        # create param histogram
        param_hist = wandb.plot.histogram(
            table,
            "draws",
            title=f"param_{idx} histogram",
        )

        # create param trace
        param_trace = wandb.plot.line(
            table,
            x="grad_evals",
            y="draws",
            title=f"param_{idx} trace",
        )

        # create se1 lineplot
        se1_lineplot = wandb.plot.line(
            table,
            x="grad_evals",
            y="se1",
            title=f"param_{idx} se1",
        )

        # create se2 lineplot
        se2_lineplot = wandb.plot.line(
            table,
            x="grad_evals",
            y="se2",
            title=f"param_{idx} se2",
        )

        # # create se1/grad table
        # se1_grad = se1[-1, idx] / grad_evals
        # se1_grad_table = wandb.Table(data=se1_grad, columns=["se1_grad"])

        # # create se2/grad table
        # se2_grad = se2[-1, idx] / grad_evals
        # se2_grad_table = wandb.Table(data=se2_grad, columns=["se2_grad"])

        # log all plots
        wandb.log(
            {
                f"param_{idx}/hist": param_hist,
                f"param_{idx}/trace": param_trace,
                f"param_{idx}/se1": se1_lineplot,
                f"param_{idx}/se2": se2_lineplot,
                f"param_{idx}/se1_grad": se1[-1, idx] / grad_evals[-1],
                f"param_{idx}/se2_grad": se2[-1, idx] / grad_evals[-1],
            }
        )


def log3(draws, grad_evals, se1, se2, step_sizes, step_counts, metric, jump_dist):
    num_draws, num_params = draws.shape

    wandb.define_metric("grad_evals", summary="last")

    for i in range(num_draws):
        if i == num_draws - 1:
            wandb.define_metric("se1_grad", summary="max")
            wandb.define_metric("se2_grad", summary="max")

        for j in range(num_params):
            # define metrics
            if i == 0:
                wandb.define_metric(
                    f"param_{j}/se1", summary="last", step_metric="grad_evals"
                )
                wandb.define_metric(
                    f"param_{j}/se2", summary="last", step_metric="grad_evals"
                )

            # log metrics
            wandb.log(
                {
                    "grad_evals": grad_evals[i],
                    f"param_{j}/se1": se1[i, j],
                    f"param_{j}/se2": se2[i, j],
                    f"param_{j}/draws": draws[i, j],
                }
            )

            if i == num_draws - 1:
                wandb.log(
                    {
                        f"param_{j}/se1_grad": se1[i, j] / grad_evals[i],
                        f"param_{j}/se2_grad": se2[i, j] / grad_evals[i],
                        "se1_grad": se1[i, j] / grad_evals[i],
                        "se2_grad": se2[i, j] / grad_evals[i],
                    }
                )

    jump_table = wandb.Table(columns=["jump_dist"], data=jump_dist)
    nuts_step_count_table = wandb.Table(
        columns=["step_counts"],
        data=step_counts.reshape(-1, 1),
    )
    nuts_metric_table = wandb.Table(
        columns=["metric"],
        data=metric.reshape(-1, 1),
    )

    wandb.log(
        {
            "jump_dist/jump_dist": jump_table,
            "nuts/step_counts": nuts_step_count_table,
            "nuts/step_size": step_sizes[-1],
            "nuts/metric": nuts_metric_table,
        }
    )

def log5(draws, config, grad_evals, se1, se2, rse1, rse2, step_sizes, step_counts, metric, jump_dist, uturns):
    num_draws, num_params = draws.shape
    
    # define wandb metrics
    wandb.define_metric("grad_evals", summary="max")
    wandb.define_metric("max_rse1", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_rse2", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_se1", step_metric="grad_evals", summary="last")
    wandb.define_metric("max_se2", step_metric="grad_evals", summary="last")
    for i in range(num_params):
        wandb.define_metric(f"param_{i}/se1_grad", summary="last")
        wandb.define_metric(f"param_{i}/se2_grad", summary="last")
        wandb.define_metric(f"param_{i}/se1", summary="last")
        wandb.define_metric(f"param_{i}/se2", summary="last")
        wandb.define_metric(f"param_{i}/rse1_grad", summary="last")
        wandb.define_metric(f"param_{i}/rse2_grad", summary="last")
        wandb.define_metric(f"param_{i}/rse1", summary="last")
        wandb.define_metric(f"param_{i}/rse2", summary="last")


    log_freq = max(1, config.grad_evals // 10000)
    for idx in range(0, len(draws), log_freq):
        wandb.log(
            {
                "grad_evals": grad_evals[idx],
                "max_se1": max(se1[idx]),
                "max_se2": max(se2[idx]),
                "max_rse1": max(rse1[idx]),
                "max_rse2": max(rse2[idx]),
            }
        )

    # for each model parameter, log squared error and squared error per gradient evaluation
    total_grad_evals = grad_evals[-1]
    for i in range(num_params):
        wandb.log(
            {
                # log squared error and squared error/gradient for each model parameter
                f"param_{i}/se1_grad": se1[-1][i] / total_grad_evals,
                f"param_{i}/se2_grad": se2[-1][i] / total_grad_evals,
                f"param_{i}/se1": se1[-1][i],
                f"param_{i}/se2": se2[-1][i],
                f"param_{i}/rse1_grad": rse1[-1][i] / total_grad_evals,
                f"param_{i}/rse2_grad": rse2[-1][i] / total_grad_evals,
                f"param_{i}/rse1": rse1[-1][i],
                f"param_{i}/rse2": rse2[-1][i],
            }
        )
        
    wandb.log(
        {
            # log final squared error
            "grad_evals": grad_evals[-1],
            "max_se1": max(se1[-1]),
            "max_se1": max(se2[-1]),
            "max_rse1": max(rse1[-1]),
            "max_rse1": max(rse2[-1]),
            # log squared error and squared error per gradient evaluation as scalars
            "squared_error/se1": max(se1[-1]),
            "squared_error/se2": max(se2[-1]),
            "squared_error/se1_grad": max(se1[-1]) / total_grad_evals,
            "squared_error/se2_grad": max(se2[-1]) / total_grad_evals,
            "squared_error/rse1": max(rse1[-1]),
            "squared_error/rse2": max(rse2[-1]),
            "squared_error/rse1_grad": max(rse1[-1]) / total_grad_evals,
            "squared_error/rse2_grad": max(rse2[-1]) / total_grad_evals,
            # log diagnostics
            "diagnostics/num_draws": num_draws,
            # log nuts params
            "nuts/step_size": step_sizes[-1],
        }
    )

    # downsample and create wandb tables
    table_dict = dict()
    if jump_dist:
        jump_dist = jump_dist[::math.ceil(len(jump_dist) / 10000)]
        jump_table = wandb.Table(columns=["jump_dist"], data=jump_dist)
        table_dict["jump_dist/jump_dist"] = jump_table
    if uturns:
        uturns = uturns[::math.ceil(len(uturns) / 10000)]
        uturns_table = wandb.Table(columns=["uturns"], data=uturns)
        table_dict["uturns/uturns"] = uturns_table

    nuts_step_count_table = wandb.Table(
        columns=["step_counts"],
        data=step_counts.reshape(-1, 1),
    )
    table_dict["nuts/step_counts"] = nuts_step_count_table

    nuts_metric_table = wandb.Table(
        columns=["metric"],
        data=metric.reshape(-1, 1),
    )
    table_dict["nuts/metric"] = nuts_metric_table

    wandb.log(table_dict)


def log4(draws, grad_evals, se1, se2, step_sizes, step_counts, uturns):
    # create wandb table for draws
    num_draws, num_params = draws.shape
    data = np.concatenate(
        (
            draws,
            se1,
            se2,
            grad_evals.reshape(-1, 1),
        ),
        axis=1,
    )
    col_names = (
        [f"param_{i}" for i in range(num_params)]
        + [f"se1_param_{i}" for i in range(num_params)]
        + [f"se2_param_{i}" for i in range(num_params)]
        + ["grad_evals"]
    )
    table = wandb.Table(data=data, columns=col_names)
    
    for idx in range(num_params):
        # plot trace plot
        trace_plot = wandb.plot.line(
            table,
            x="grad_evals",
            y=f"param_{idx}",
            title=f"param_{idx} trace",
        )
        
        # plot se1 plot
        se1_plot = wandb.plot.line(
            table,
            x="grad_evals",
            y=f"se1_param_{idx}",
            title=f"param_{idx} se1",
        )
        
        # plot se2 plot
        se2_plot = wandb.plot.line(
            table,
            x="grad_evals",
            y=f"se2_param_{idx}",
            title=f"param_{idx} se2",
        )
        
        # log plots
        wandb.log(
            {
                f"param_{idx}/trace": trace_plot,
                f"param_{idx}/se1": se1_plot,
                f"param_{idx}/se2": se2_plot,
            }
        )
        
    # log step sizes and step counts
    nuts_table = wandb.Table(
        columns=["step_counts"],
        data=step_counts.reshape(-1, 1),
    )
    wandb.log(
        {
            "nuts/step_counts": nuts_table,
            "nuts/step_size": step_sizes[-1],
        }
    )


def main(config):
    model, data, seed, init, inv_metric = configure_sampler(config)
    nuts = generate_samples(model, data, seed, init, inv_metric)

    draws, grad_evals, se1, se2, rse1, rse2, step_sizes, step_counts, metric, jumps, uturns = process_nuts(
        nuts, config
    )
    log5(draws, config, grad_evals, se1, se2, rse1, rse2, step_sizes, step_counts, metric, jumps, uturns)


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

    args = parser.parse_args()

    irrelevant_hyperparams = [
        "posterior",
        "posterior_dir",
        "seed",
        "sampler_type",
        "burn_in",
        "grad_evals",
    ]

    group = "__".join(
        [args.sampler_type]
        + [
            f"{k}-{v}"
            for k, v in args.__dict__.items()
            if k not in irrelevant_hyperparams and k not in ["chain"]
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
        name=run_name,
        group=group,
        job_type=args.sampler_type,
        save_code=False,
    )
    main(wandb.config)
