import argparse
import asyncio
import csv
import itertools
import json
import nest_asyncio
import os
import zipfile

import numpy as np
import pandas as pd
import polars as pl
from posteriordb import PosteriorDatabase


def get_nuts_path(data_path):
    nuts_path = None
    for dir in os.listdir(data_path):
        if dir.startswith("nuts"):
            nuts_path = os.path.join(data_path, dir, "chain_00")
            break

    if not nuts_path:
        raise ValueError("No NUTS directory found in {}".format(data_path))
    return nuts_path


def get_number_of_columns(csv_file_path):
    """
    Get the number of columns in a tab-separated CSV file.

    Args:
        csv_file_path (str): Path to the tab-separated CSV file.

    Returns:
        int: Number of columns in the CSV file.
    """
    num_columns = 0

    try:
        with open(csv_file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter="\t")
            first_row = next(csv_reader, None)
            if first_row:
                num_columns = len(first_row)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return num_columns


def create_nuts_df(data_path):
    # Initialize an empty list to store data frames from each run
    nuts_path = get_nuts_path(data_path)
    nuts_list = []
    num_columns = None

    # Iterate through the run directories
    for run_dir in os.listdir(nuts_path):
        if run_dir.startswith("run_"):
            run_number = int(run_dir.split("_")[1])
            csv_path = os.path.join(nuts_path, run_dir, "draws.csv")

            if os.path.exists(csv_path):
                # Add a "run" column with the run number
                df = pd.read_csv(
                    csv_path, sep="\t", usecols=["stepsize__", "n_leapfrog__"]
                )
                df["run"] = run_number

                # Get the total number of columns in the data frame
                if not num_columns:
                    num_columns = get_number_of_columns(csv_path)

                # Add the remaining columns by index starting from 11 (0-based)
                remaining_columns = list(range(11, num_columns))
                new_col_names = [f"p{idx}" for idx, val in enumerate(remaining_columns)]

                # Read the CSV file again for the remaining columns
                remaining_df = pd.read_csv(
                    csv_path, sep="\t", usecols=remaining_columns
                )

                # Combine the dataframes by concatenating them
                combined_df = pd.concat([df, remaining_df], axis=1)
                nuts_list.append(combined_df)

    # Concatenate all data frames into a single data frame
    nuts_df = pd.concat(nuts_list, ignore_index=True)
    return nuts_df


def process_nuts_df(nuts_df):
    # Step 1: Rename columns
    nuts_df.rename(
        columns={"stepsize__": "stepsize", "n_leapfrog__": "stepcount"}, inplace=True
    )

    # Step 2: Rename arbitrary columns as p1, p2, etc.
    for i in range(3, len(nuts_df.columns)):
        nuts_df.rename(columns={nuts_df.columns[i]: f"p{i-3}"}, inplace=True)

    # Step 3: Convert 'run' and 'stepcount' to unsigned int with minimal precision
    nuts_df["run"] = pd.to_numeric(nuts_df["run"], downcast="unsigned")
    nuts_df["stepcount"] = pd.to_numeric(nuts_df["stepcount"], downcast="unsigned")

    # Step 4: Convert remaining columns to the smallest float data type
    for col in nuts_df.columns[3:]:
        nuts_df[col] = pd.to_numeric(nuts_df[col], downcast="float")

    # Step 5: Reorder columns with 'run' as the first column, 'stepsize' as the second column, and 'stepcount' as the third column
    column_order = ["run", "stepsize", "stepcount"] + list(
        nuts_df.columns.difference(["run", "stepsize", "stepcount"])
    )
    nuts_df = nuts_df[column_order]

    # Step 6: Add a new column "sampler" with the value "NUTS" for every row
    nuts_df.insert(0, "sampler", "nuts")

    # convert "stepsize" column to float32 in Pandas dataframe
    nuts_df = nuts_df.astype({"stepsize": "float32"})
    nuts_df = pl.from_pandas(nuts_df)

    return nuts_df


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def load_draws(file_path):
    draws_array = np.load(file_path)
    draws_df = pl.from_numpy(draws_array)

    # rename columns as "p1", "p2", etc.
    draws_df = draws_df.rename({c: f"p{idx}" for idx, c in enumerate(draws_df.columns)})
    return draws_df


def load_sampler_params(file_path, num_draws):
    sampler_params_df = pl.read_json(file_path)

    sampler_params_df = sampler_params_df.select(
        [
            "init_stepsize",
            "reduction_factor",
            "steps",
            "dampening",
            "num_proposals",
            "probabilistic",
            "sampler_type",
            "grad_evals",
        ]
    )

    schema = {
        "init_stepsize": pl.Float32,
        "reduction_factor": pl.UInt8,
        "steps": pl.Utf8,
        "dampening": pl.Float32,
        "num_proposals": pl.UInt8,
        "probabilistic": pl.Boolean,
        "grad_evals": pl.UInt32,
    }

    sampler_params_df = sampler_params_df.with_columns(
        pl.col(c).cast(dtype) for c, dtype in schema.items()
    )

    # Use concat to vertically stack the original DataFrame n times
    sampler_params_repeated = pl.concat([sampler_params_df for _ in range(num_draws)])

    return sampler_params_repeated


def load_hyper_params(file_path, num_draws):
    hyper_params_df = pl.read_json(file_path)

    hyper_params_df = hyper_params_df.select("global_seed").rename(
        {"global_seed": "run"}
    )

    schema = {"run": pl.UInt8}
    hyper_params_df = hyper_params_df.with_columns(
        pl.col(c).cast(dtype) for c, dtype in schema.items()
    )

    hyper_params_repeated = pl.concat([hyper_params_df for _ in range(num_draws)])

    return hyper_params_repeated


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def helper(root, dir_name):
    data = []

    if dir_name.startswith("drhmc") or dir_name.startswith("drghmc"):
        for run_dir in os.listdir(os.path.join(root, dir_name, "chain_00")):
            if run_dir.startswith("run_"):
                draws_path = os.path.join(
                    root, dir_name, "chain_00", run_dir, "draws.npy"
                )
                sampler_params_path = os.path.join(
                    root, dir_name, "chain_00", run_dir, "sampler_params.json"
                )
                hyper_params_path = os.path.join(
                    root, dir_name, "chain_00", run_dir, "hyper_params.json"
                )

                if (
                    os.path.exists(draws_path)
                    and os.path.exists(sampler_params_path)
                    and os.path.exists(hyper_params_path)
                ):
                    # Step 1
                    draws_df = load_draws(draws_path)
                    num_draws = draws_df.shape[0]

                    # Step 2
                    sampler_params_df = load_sampler_params(
                        sampler_params_path, num_draws
                    )

                    # Step 3
                    hyper_params_df = load_hyper_params(hyper_params_path, num_draws)

                    # Combine dataframes by repeating sampler_params and hyper_params for each row in draws_df
                    combined_df = pl.concat(
                        [hyper_params_df, sampler_params_df, draws_df], how="horizontal"
                    )

                    data.append(combined_df)
    return data


def samples_to_polars_df(data_path):
    data = []
    root, dirs, _ = next(os.walk(data_path))

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    looper = asyncio.gather(*[helper(root, dir_name) for dir_name in dirs])
    data = loop.run_until_complete(looper)
    data = list(itertools.chain.from_iterable(data))

    samples = pl.concat(data)
    samples = samples.rename(
        {
            "sampler_type": "sampler",
            "init_stepsize": "stepsize",
            "steps": "stepcount",
        }
    )

    # if "stepcount" column contains "const_traj_len", convert it to 0
    samples = samples.with_columns(
        pl.when(pl.col("stepcount") == "const_traj_len")
        .then(0)
        .otherwise(pl.col("stepcount"))
        .alias("stepcount")
    )

    samples = samples.with_columns(pl.col("stepcount").cast(pl.UInt16))

    return samples


def get_ref_draws_json(posterior_name, posterior_path):
    try:  # load posterior from custom model
        path = os.path.join(posterior_path, posterior_name)
        ref_draws_path = os.path.join(path, f"{posterior_name}.ref_draws.json.zip")
        ref_draws = json.load(
            zipfile.ZipFile(ref_draws_path).open(f"{posterior_name}.ref_draws.json")
        )

    except:  # try to load posterior from PDB
        path = os.path.join(posterior_path, "posteriordb/posterior_database")
        pdb = PosteriorDatabase(path)
        posterior = pdb.posterior(posterior_name)
        ref_draws = posterior.reference_draws()

    return ref_draws


def json_to_polars(ref_draws):
    data = []
    for idx, run in enumerate(ref_draws):
        cur_df = pl.DataFrame(run)
        cur_df = cur_df.rename({c: f"p{idx}" for idx, c in enumerate(cur_df.columns)})

        cur_df = cur_df.with_columns(pl.col(c).cast(pl.Float32) for c in cur_df.columns)
        cur_df = cur_df.with_columns(pl.lit(idx).alias("run"))
        cur_df = cur_df.with_columns(pl.col("run").cast(pl.UInt8))

        data.append(cur_df)

    # return Polars dataframe and add a column called "sampler" with value "reference"
    return pl.concat(data).with_columns(pl.lit("ref").alias("sampler"))


def main(posterior_name, raw_data_path, posterior_path):
    dr_samplers = samples_to_polars_df(raw_data_path)
    nuts = process_nuts_df(create_nuts_df(raw_data_path))
    ref_draws = json_to_polars(get_ref_draws_json(posterior_name, posterior_path))

    # Combine the two dataframes
    samples = pl.concat([dr_samplers, nuts, ref_draws], how="diagonal")
    samples = samples.sort("run")

    # convert the "sampler" column to categorical
    samples = samples.with_columns(pl.col("sampler").cast(pl.Categorical))

    samples = samples.select(
        ["run", "sampler", "stepcount"]
        + diff(samples.columns, ["run", "sampler", "stepcount"])
    )

    # save samples dataframe in compact format

    path = os.path.join("data/processed/", posterior_name, "samples.parquet")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    samples.write_parquet(path, compression_level=22)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior_name", type=str, help="PDB model number")
    args = parser.parse_args()
    
    posterior_name = args.posterior_name
    raw_data_path = os.path.join("data", "raw", posterior_name)
    posterior_path = os.path.join("posteriors")

    main(posterior_name, raw_data_path, posterior_path)
