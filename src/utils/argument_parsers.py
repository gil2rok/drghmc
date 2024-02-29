import argparse


def summarize_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="experiment name used for saving data, loading data, and selecting Weights and Biases project",
    )
    parser.add_argument(
        "--posterior",
        type=str,
        required=True,
        help="name of posterior to sample from",
    )
    parser.add_argument(
        "--posterior_dir",
        type=str,
        required=True,
        help="directory to posterior",
    )
    return parser


def drhmc_argument_parser():
    parser = _base_sampler_argument_parser()
    parser.add_argument(
        "--sampler_type",
        type=str,
        required=True,
        help="type of sampling algorithm to use",
    )
    parser.add_argument(
        "--chain",
        type=int,
        required=True,
        help="chain number",
    )
    step_count_group = parser.add_mutually_exclusive_group(required=True)
    step_count_group.add_argument(
        "--step_count",
        type=int,
        help="number of leapfrog steps for first proposal",
    )
    step_count_group.add_argument(
        "--step_count_factor",
        type=float,
        help="number of leapfrog steps for first proposal, computed as the NUTS step count multiplied by this factor",
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
        required=True,
        help="maximum number of proposals to make",
    )
    parser.add_argument(
        "--reduction_factor",
        type=float,
        required=True,
        help="factor by which to reduce the step size in subsequent proposals",
    )
    parser.add_argument(
        "--damping",
        type=float,
        required=True,
        help="damping parameter for momentum refresh in generalized HMC",
    )
    parser.add_argument(
        "--metric",
        type=int,
        required=True,
        help="metric type to use for preconditioning",
    )
    parser.add_argument(
        "--probabilistic",
        help="whether to use probabilistic retry",
        action="store_true",
    )
    return parser
        

def drghmc_argument_parser():
    parser = _base_sampler_argument_parser()
    parser.add_argument(
        "--sampler_type",
        type=str,
        required=True,
        help="type of sampling algorithm to use",
    )
    parser.add_argument(
        "--chain",
        type=int,
        required=True,
        help="chain number",
    )
    parser.add_argument(
        "--step_count_method",
        type=str,
        required=True,
        choices=["const_step_count", "const_traj_length"],
        help="method to compute number of leapfrog steps in subsequent proposals. As a generalized HMC method, the first step count is always one. Can specificy a constant trajectory length that increases the step count in subsequent proposals, or a constant step count that uses the same step count (of one) for all proposals",
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
        required=True,
        help="maximum number of proposals to make",
    )
    parser.add_argument(
        "--reduction_factor",
        type=float,
        required=True,
        help="factor by which to reduce the step size in subsequent proposals",
    )
    parser.add_argument(
        "--damping",
        type=float,
        required=True,
        help="damping parameter for momentum refresh in generalized HMC",
    )
    parser.add_argument(
        "--metric",
        type=int,
        required=True,
        help="metric type to use for preconditioning",
    )
    parser.add_argument(
        "--probabilistic",
        help="whether to use probabilistic retry",
        action="store_true",
    )
    return parser


def nuts_argument_parser():
    parser = _base_sampler_argument_parser()
    parser.add_argument(
        "--sampler_type",
        type=str,
        required=True,
        help="type of sampling algorithm to use",
    )
    parser.add_argument(
        "--chain",
        type=int,
        required=True,
        help="chain number",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["identity", "diag_cov"],
        help="metric type to use for preconditioning. Can be identity or diagonal covariance (derived from reference draws)",
    )
    return parser


def _base_sampler_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="experiment name used for saving data, loading data, and selecting Weights and Biases project",
    )
    parser.add_argument(
        "--posterior",
        type=str,
        required=True,
        help="name of posterior to sample from",
    )
    parser.add_argument(
        "--posterior_dir",
        type=str,
        required=True,
        help="directory to posterior",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="global random seed for experiment",
    )
    parser.add_argument(
        "--generate_history",
        action="store_true",
        help="record sampler history: draws, gradient evaluations, acceptance probablities, number of nans, leapfrog step sizes, and leapfrog step counts",
    )
    parser.add_argument(
        "--generate_metrics",
        action="store_true",
        help="record sampler metrics: squared error, jump distance, and u-turn distance",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        required=True,
        help="number of burn-in iterations",
    )
    parser.add_argument(
        "--gradient_budget",
        type=int,
        required=True,
        help="number of approximate gradient evaluations to run the sampling algorithm for",
    )
    return parser