import argparse
import asyncio
import itertools
import json
import nest_asyncio
import os
from zipfile import ZipFile

import polars as pl
from posteriordb import PosteriorDatabase

from .utils.summary_stats import get_summary_stats


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def helper(root, dir_name):
    data = []
    if (
        dir_name.startswith("drhmc")
        or dir_name.startswith("drghmc")
        or dir_name.startswith("ghmc")
    ):
        chain_dir = os.path.join(root, dir_name)
        _, chains, _ = next(os.walk(chain_dir))
        chains.sort()

        for idx, chain in enumerate(chains):
            chain_path = os.path.join(chain_dir, chain)
            
            params = json.load(open(os.path.join(chain_path, "params.json")))

            summary_stats_packed = json.load(
                open(os.path.join(chain_path, "summary_stats.json"))
            )
            sampler_params = json.load(
                open(os.path.join(chain_dir, "sampler_params.json"))
            )

            summary_stats = {}
            for k, v in summary_stats_packed.items():
                for param_idx, param in enumerate(v):
                    new_key = k + f"_p{param_idx}"
                    new_val = summary_stats_packed[k][param_idx]  # is this equivalent to new_val = param?
                    summary_stats[new_key] = new_val

            sampler_id = {"sampler": dir_name, "chain": idx}
            row = sampler_id | sampler_params | params | summary_stats
            data.append(row)
    return data


def get_bk_df(data_path, posterior):
    data = []
    root, dirs, _ = next(os.walk(os.path.join(data_path, "raw", posterior)))

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    looper = asyncio.gather(*[helper(root, dir_name) for dir_name in dirs])
    data = loop.run_until_complete(looper)
    data = list(itertools.chain.from_iterable(data))

    samples = pl.DataFrame(data)
    return samples


def get_nuts_df(data_path, posterior):
    nuts_path = os.path.join(data_path, "raw", posterior, "nuts")
    _, chains, _ = next(os.walk(nuts_path))

    data = []
    for idx, chain in enumerate(chains):
        chain_path = os.path.join(nuts_path, chain)
        
        params = json.load(open(os.path.join(chain_path, "params.json")))
        params.pop("inv_metric", None)

        summary_stats_packed = json.load(
            open(os.path.join(chain_path, "summary_stats.json"))
        )

        summary_stats = {}
        for k, v in summary_stats_packed.items():
            for param_idx, param in enumerate(v):
                new_key = k + f"_p{param_idx}"
                new_val = summary_stats_packed[k][param_idx]
                summary_stats[new_key] = new_val

        sampler_id = {"sampler": "nuts", "chain": idx}
        row = sampler_id | params | summary_stats
        data.append(row)
    return pl.DataFrame(data)


def get_ref_draws(posterior_path, posterior_name):
    """Returns list of dictionaries, where each dict represents an individual chain.
    Each dict has keys as parameter names and values as a list of parameter draws.
    """
    try:  # try to load posterior from PDB
        path = os.path.join(posterior_path, "posteriordb/posterior_database")
        pdb = PosteriorDatabase(path)
        posterior = pdb.get_posterior(posterior_name)
        ref_draws = posterior.reference_draws()

    except:  #  load posterior from custom model
        path = os.path.join(posterior_path, posterior_name)
        ref_draws_path = os.path.join(path, f"{posterior_name}.ref_draws.json.zip")
        ref_draws = json.loads(
            ZipFile(ref_draws_path)
            .read(f"{posterior_name}.ref_draws.json")
            .decode("utf-8")
        )

    return ref_draws


def get_ref_df(posterior_path, posterior_name):
    data = []
    ref_draws = get_ref_draws(
        posterior_path, posterior_name
    )  # [num_chains  x num_params]

    for idx, chain in enumerate(ref_draws):
        summary_stats = {}
        for param_idx, params in enumerate(chain.values()):
            summary_stats_packed = get_summary_stats(params)

            for k, v in summary_stats_packed.items():
                new_key = k + f"_p{param_idx}"
                summary_stats[new_key] = v

        sampler_id = {"sampler": "ref", "sampler_type": "ref", "chain": idx}
        row = sampler_id | summary_stats
        data.append(row)

    return pl.DataFrame(data)


def merge_dataframes(bk_df, nuts_df, ref_df):
    samples_df = pl.concat([bk_df, nuts_df, ref_df], how="diagonal")

    schema = {
        "sampler": pl.Categorical,
        "chain": pl.UInt8,
        "sampler_type": pl.Categorical,
        "init_stepsize": pl.Float32,
        "reduction_factor": pl.UInt8,
        "steps": pl.Utf8,
        "dampening": pl.Float32,
        "num_proposals": pl.UInt8,
        "probabilistic": pl.Boolean,
        "grad_evals": pl.UInt32,
    }

    samples_df = samples_df.with_columns(
        pl.col(c).cast(dtype) for c, dtype in schema.items()
    )

    return samples_df


def main(posterior, data_path, posterior_path):
    """ Create dataframe containing summary statistics of reference draws, NUTS draws, 
    and Bayes-Kit sampler draws (e.g. GHMC, DRHMC, and DRGHMC).
    
    These summary statistics include the mean and squared mean of every parameter, 
    as well as the and number of gradient evaluations. Crucially, these summary 
    statistics are computed only from the draws and monitoring the samplers that 
    generate them.

    Args:
        posterior: name of posterior
        data_path: path to data directory containing draws
        posterior_path: path to posterior directory containing reference draws

    Returns:
        samples_df: dataframe containing mean, squared mean, and gradient evaluations 
        for reference draws, draws from NUTS, and draws from Bayes-Kit samplers
    """
    
    bk_df = get_bk_df(data_path, posterior)
    nuts_df = get_nuts_df(data_path, posterior)
    ref_df = get_ref_df(posterior_path, posterior)

    samples_df = merge_dataframes(bk_df, nuts_df, ref_df)

    save_path = os.path.join(data_path, "processed", posterior, "samples.parquet")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    samples_df.write_parquet(save_path, compression_level=22)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posterior",
        type=str,
        help="name of a posterior, specified by a Stan model and data",
    )
    args = parser.parse_args()

    posterior = args.posterior
    data_path = os.path.join("data")
    posterior_path = os.path.join("posteriors")

    main(posterior, data_path, posterior_path)
