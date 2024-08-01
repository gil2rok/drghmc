import os

import hydra
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
import seaborn as sns

from src.utils.summary_utils import read_from_summary

sns.set_theme(style="whitegrid", font_scale=1.8)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
})


def load_data(config):
    path = os.path.join("data", config.posterior.name, "summary.csv")
    summary_lazy = pl.scan_csv(path, has_header=True)
    # cast a few columns to specifc polars data types
    summary_lazy = summary_lazy.with_columns(
        pl.col("step_size").cast(pl.Float32),
        pl.col("step_size_factor").cast(pl.Float32),
        pl.col("step_count").cast(pl.Int32),
        pl.col("step_count_factor").cast(pl.Float32),
        pl.col("max_proposals").cast(pl.Int32),
        pl.col("reduction_factor").cast(pl.Float32),
        pl.col("damping").cast(pl.Float32),
        pl.col("adapt_metric").cast(pl.Boolean),
    )

    # filter by the tags and then sort according to the hyper_param
    summary = (
        summary_lazy.filter((pl.col("tags") == config.tags))
        .sort(
            pl.col(config.hyper_param),
        )
        .collect()
    )

    if config.tags == "baseline" and config.posterior.name not in set(["funnel10", "funnel30", "funnel50"]):
        print("HERE")
        summary = summary.filter(
            ~pl.col("group").str.contains(
                "adapt_metric=False__metric=identity__sampler_type=nuts"
            )
        )

    summary = summary.with_columns(
        pl.col("sampler_type").replace("nuts", "NUTS").replace("drhmc", "DR-HMC").replace("drghmc", "DR-G-HMC")
    )
    return summary


def setup(summary, config):
    hyper_params_order = summary[config.hyper_param].unique().sort().to_numpy()
    # if config.posterior.name == "funnel10":
    #     error_params = [
    #         "error_log_scale",
    #         "error_log_scale_squared",
    #         "error_latent",
    #         "error_latent_squared",
    #     ]
    # else:
    #     error_params = ["error_param", "error_param_squared"]
    error_params = ["error_param", "error_param_squared"]

    return hyper_params_order, error_params


def save_to_csv(summary, error_params, hyper_param, figures_dir):
    if error_params != ["error_param", "error_param_squared"]:
        error_params += ["error_param", "error_param_squared"]

    for c in error_params:
        grouped = (
            summary.group_by("group")
            .agg(
                pl.first("sampler_type").alias("sampler_type"),
                (
                    pl.first(hyper_param).alias(hyper_param)
                    if hyper_param != "sampler_type"
                    else None
                ),
                pl.mean(c).alias("mean"),
                pl.std(c).alias("std"),
                pl.median(c).alias("median"),
                pl.quantile(c, 0.1).alias("10"),
                pl.quantile(c, 0.9).alias("90"),
            )
            .sort(hyper_param)
        )

        # add column named "error" with value c
        grouped = grouped.with_columns(pl.lit(c).alias("error"))

        fname = os.path.join(figures_dir, f"normalized_error.csv")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "a") as f:
            include_header = c == error_params[0]
            grouped.write_csv(f, include_header=include_header)

        markdown = grouped.to_pandas().to_markdown(tablefmt="simple_outline")
        print(c, "\n", markdown, "\n\n")


def make_box_plot(data, hyper_param, posterior, value_vars, figures_dir):
    fig = sns.catplot(
        data=data,
        kind="box",
        x=hyper_param,
        y="normalized error",
        hue="sampler_type",
        # hue_order=hyper_param_order,
        col="param",
        col_wrap=2,
        col_order=value_vars,
        aspect=2,
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="--", linewidth=2, color="black"),
    )

    # set y axis to log scale
    fig.set(yscale="log")
    fig.axes.flat[0].set_ylabel("Error in Mean")
    fig.axes.flat[1].set_ylabel("Error in Variance")
    
    fname = os.path.join(figures_dir, f"box_error.png")
    fig.savefig(fname)


def make_mean_plot(data, hyper_param, posterior, value_vars, figures_dir):
    fig = sns.catplot(
        data=data,
        kind="point",
        x=hyper_param,
        y="normalized error",
        hue="sampler_type",
        estimator=np.mean,
        errorbar=None,
        col="param",
        col_wrap=2,
        col_order=value_vars,
        aspect=1.5,
    )

    fig.set(yscale="log")
    fname = os.path.join(figures_dir, f"mean_error.png")
    fig.savefig(fname)


def make_mean_median_plot(
    data, hyper_params_order, hyper_param, posterior, value_vars, figures_dir
):
    g = sns.FacetGrid(
        data=data,
        hue="sampler_type",
        col="param",
        col_wrap=2,
        col_order=value_vars,
        aspect=1.5,
        height=5,
    )
    g.map(
        sns.pointplot,
        hyper_param,
        "normalized error",
        estimator=np.mean,
        errorbar=None,
        markers="o",
        order=hyper_params_order,
    )
    g.map(
        sns.pointplot,
        hyper_param,
        "normalized error",
        estimator=np.median,
        errorbar=None,
        markers="^",
        order=hyper_params_order,
        err_kws={"linestyle": "-"},
        linestyle="--",
    )
    g.set(yscale="log")

    hues = data["sampler_type"].unique()
    custom_legend = [
        Line2D([0], [0], color=sns.color_palette()[i], lw=2, label=hue)
        for i, hue in enumerate(hues)
    ]
    custom_legend += [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            markersize=10,
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="black",
            markerfacecolor="black",
            markersize=10,
            label="Median",
            linestyle="--",
        ),
    ]

    plt.legend(
        handles=custom_legend,
        loc="upper right",
        bbox_to_anchor=(1.2, 0.5),
        ncol=1,
        title="Sampler",
    )  # legend to the right of figure
    # plt.legend(handles=custom_legend, loc="lower center", bbox_to_anchor=(0, -0.3), ncol=3) # legend below figure
    fname = os.path.join(figures_dir, f"mean_median_error.png")
    g.savefig(fname)


def make_grad_vs_error_plot(
    summary, hyper_params_order, error_params, hyper_param, posterior, figures_dir
):
    history_list = ["grad_evals"]
    metrics_list = error_params
    n_samples = 100000
    complete_summary = read_from_summary(summary, history_list, metrics_list, n_samples)

    grad_rounding = -3
    complete_summary = complete_summary.with_columns(
        pl.col("grad_evals")
        .map_elements(lambda s: np.round(s, grad_rounding))
        .alias("gradient evaluations")
    )

    value_vars = error_params
    id_vars = [col for col in complete_summary.columns if col not in value_vars]
    melted = complete_summary.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        variable_name="params",
        value_name="normalized error",
    )
    data = melted.to_pandas()

    fig = sns.relplot(
        data=data,
        kind="line",
        x="gradient evaluations",
        y="normalized error",
        hue=hyper_param,
        hue_order=hyper_params_order,
        style="sampler_type" if hyper_param != "sampler_type" else None,
        errorbar=None,
        estimator="mean",
        col="params",
        col_wrap=2,
        col_order=value_vars,
        aspect=1.5,
        height=5,
        # facet_kws={"sharey": False},
    )


    fig.set(yscale="log")        
    # set title for each subplot
    fig.axes.flat[0].set_title(r'Avg Error in Mean ($\mathcal{L}_{\theta, T}$)')
    fig.axes.flat[1].set_title(r'Avg Error in Variance ($\mathcal{L}_{\theta^2, T}$)')
    
    fig.set_ylabels("Error")
    # fig.set_xlabels(r'$t$ Gradient Evaluations')
    fig.set_xlabels(r'Num Gradient Evaluations')
    fig.figure.subplots_adjust(bottom=0.3)
    sns.move_legend(fig, "lower center", ncol=3, title="")
    fname = os.path.join(figures_dir, f"grad_vs_error")
    fig.savefig(fname)


@hydra.main(version_base=None, config_path="../configs/", config_name="figures")
def main(config):

    summary = load_data(config)
    hyper_params_order, error_params = setup(summary, config)
    save_to_csv(summary, error_params, config.hyper_param, config.figures.dir)

    id_vars = [col for col in summary.columns if col not in error_params]
    melted = summary.melt(
        id_vars=id_vars,
        value_vars=error_params,
        variable_name="param",
        value_name="normalized error",
    )
    data = pd.DataFrame(melted.to_pandas())

    make_box_plot(
        data,
        config.hyper_param,
        config.posterior.name,
        error_params,
        config.figures.dir,
    )
    make_mean_plot(
        data,
        config.hyper_param,
        config.posterior.name,
        error_params,
        config.figures.dir,
    )
    make_mean_median_plot(
        data,
        hyper_params_order,
        config.hyper_param,
        config.posterior.name,
        error_params,
        config.figures.dir,
    )
    try:
        make_grad_vs_error_plot(
            summary,
            hyper_params_order,
            error_params,
            config.hyper_param,
            config.posterior.name,
            config.figures.dir,
        )
    except Exception as e:
        print(f"Unable to create gradient vs error plot:\n {e}")


if __name__ == "__main__":
    main()
