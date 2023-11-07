from pathlib import Path
import numpy as np
import json
import os
import functools

from .posteriordb_utils import BSDB
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


def get_init(model_num, pdb_dir):
    return BSDB(model_num, pdb_dir).get_reference_draws()


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


def stan_save(nuts, sampler_type, hp):
    draws = nuts.draws()  # [num samples, num chains, num params]
    stepsize = nuts.step_size
    metric = nuts.metric
    
    param_hash = get_param_hash("", sampler_type, stepsize, metric)
    dir_name = os.path.join(
        hp.save_dir,
        hp.model_num,
        f"{sampler_type}_{param_hash}",
        f"chain_{hp.chain_num:02d}",
    )
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    np.save(os.path.join(dir_name, "draws"), draws.astype(np.float16))
    
    # save hyper parameters as json
    with open(os.path.join(dir_name, "hyper_params.json"), "w") as file:
        file.write(json.dumps(hp._asdict()))
        
    # save sampler parameters as json
    with open(os.path.join(dir_name, "sampler_params.json"), "w") as file:
        sp_dict = {}
        sp_dict["sampler_type"] = sampler_type
        sp_dict["stepsize"] = float(stepsize)
        sp_dict["metric"] = metric.tolist()[0]
        
        print(type(sp_dict["metric"]), sp_dict["metric"])
        file.write(json.dumps(sp_dict))


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
        hp.model_num,
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
