from pathlib import Path
import numpy as np
import json
import os
import functools

from .posteriordb import BSDB
from .hash_util import get_hash_str


def call_counter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper

def grad_counter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        lp, grad = f(*args, **kwargs)
        if grad is not None:
            wrapper.calls += 1
        return (lp, grad)

    wrapper.calls = 0
    return wrapper


def get_model(model_num, pdb_dir):
    return BSDB(model_num, pdb_dir)


class DualAveragingStepSize:
    def __init__(
        self,
        initial_step_size,
        target_accept=0.65,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
        nadapt=0,
    ):
        self.initial_step_size = initial_step_size
        self.mu = np.log(
            10 * initial_step_size
        )  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0
        self.nadapt = nadapt

    def update(self, p_accept):
        if np.isnan(p_accept):
            p_accept = 0.0
        if p_accept > 1:
            p_accept = 1.0
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t**-self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)

    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n" % step_size)
        else:
            step_size = np.exp(self.log_averaged_step)
        return step_size


def get_param_hash(sp, sampler_type, burn_in, chain_length):
    hash1 = get_hash_str(sp)
    hash2 = sampler_type
    hash3 = str(burn_in)
    hash4 = str(chain_length)
    return get_hash_str(hash1 + hash2 + hash3 + hash4)


def acceptance_helper(acceptances, proposal_num):
    num_proposed = np.count_nonzero(np.abs(acceptances) >= proposal_num + 1)
    num_accepted = np.count_nonzero(acceptances == proposal_num + 1)
    try:
        accept_rate = num_accepted / num_proposed
    except:
        accept_rate = -1
    return accept_rate


def compute_acceptance(sampler, sampler_type, burn_in, sp):
    if sampler_type == "bk_hmc" or sampler_type == "bk_mala":
        return None
    
    # compute acceptance statistics
    all_acceptances = np.asanyarray(sampler._acceptance_list, dtype=np.float16)
    burned_acceptances = all_acceptances[:burn_in]
    acceptances = all_acceptances[burn_in:]
    
    accept_list = []
    for i in range(sp.num_proposals):
        accept_rate = acceptance_helper(acceptances, i)
        accept_list.append(accept_rate)
    
    total_proposed = np.count_nonzero(np.abs(acceptances) >= 1)
    total_accepted = np.count_nonzero(acceptances > 0)
    accept_rate_total = total_accepted / total_proposed
    
    return acceptances, burned_acceptances, accept_list, accept_rate_total


def my_save(sp, hp, burned_draws, draws, sampler_type, sampler):
    # burn in and chain length
    burn_in = (
        int(hp.burn_in_gradeval / sp.steps)
        if type(sp.steps) is int
        else hp.burn_in_gradeval
    )
    chain_len = (
        int(hp.chain_length_gradeval / sp.steps)
        if type(sp.steps) is int
        else hp.chain_length_gradeval
    )

    # create directory
    param_hash = get_param_hash(sp, sampler_type, burn_in, chain_len)
    dir_name = os.path.join(
        hp.save_dir,
        f"PDB_{hp.model_num:02d}",
        f"{sampler_type}_{param_hash}",
        f"chain_{hp.chain_num:02d}",
    )
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    accept_tuple = compute_acceptance(sampler, sampler_type, burn_in, sp)
    
    # save burned draws, draws, and acceptances as numpy arrays
    np.save(os.path.join(dir_name, "burned_draws"), burned_draws.astype(np.float16))
    np.save(os.path.join(dir_name, "draws"), draws.astype(np.float16))
    
    if accept_tuple is not None:
        acceptances, burned_acceptances = accept_tuple[0], accept_tuple[1]
        np.save(os.path.join(dir_name, "burned_acceptances"), burned_acceptances)
        np.save(os.path.join(dir_name, "acceptances"), acceptances)

    # save hyper parameters as json
    with open(os.path.join(dir_name, "hyper_params.json"), "w") as file:
        file.write(json.dumps(hp._asdict()))

    # save sampler parameters as json
    with open(os.path.join(dir_name, "sampler_params.json"), "w") as file:
        sp_dict = sp._asdict()
        sp_dict["sampler_type"] = sampler_type
        sp_dict["burn_in"] = burn_in
        sp_dict["chain_length"] = chain_len
        sp_dict["grad_evals"] = sampler._model.log_density_gradient.calls
        sp_dict["density_evals"] = (
            sampler._model.log_density_gradient.calls + sampler._model.log_density.calls
        )
        
        if accept_tuple is not None:
            accept_list, accept_rate_total = accept_tuple[2], accept_tuple[3]
            sp_dict["accept_total"] = accept_rate_total
            for i in range(sp.num_proposals):
                sp_dict[f"accept_{i}"] = accept_list[i]
        
        file.write(json.dumps(sp_dict))
