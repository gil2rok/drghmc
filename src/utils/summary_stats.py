import numpy as np

def mean(draws):
    return np.mean(draws, axis=0).tolist()


def ess(draws, ref_draws):
    std_error = np.std(draws - ref_draws, axis=0)
    std_dev = np.std(draws, axis=0)
    return float(std_dev / std_error)**2


def get_summary_stats(draws):
    summary_stats = {
        "mean": mean(draws), 
        "mean_squared": mean( np.square(draws)),
    }
    return summary_stats
