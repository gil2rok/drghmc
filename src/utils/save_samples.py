from pathlib import Path
import numpy as np
import json
import os
import functools

import pandas as pd

from .hashing import get_hash_str


def grad_counter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


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


def stan_save(nuts, hp, summary_stats):
    save_path = os.path.join(
        hp.save_dir,
        hp.posterior,
        f"nuts",
        f"chain_{hp.chain:02d}"
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    draws_df = nuts.draws_pd()
    draws_df.to_csv(os.path.join(save_path, "draws.csv"), sep="\t", float_format="%.64f")
    
    # save hyper parameters as json
    # with open(os.path.join(dir_name, "hyper_params.json"), "w") as file:
        # file.write(json.dumps(hp._asdict()))
        
    # save sampler parameters as json
    with open(os.path.join(save_path, "params.json"), "w") as file:
        sp_dict = {
            "sampler_type": "nuts",
            "init_stepsize": float(nuts.step_size),
            "inv_metric": nuts.metric.tolist()[0],
            "grad_evals": int(draws_df["n_leapfrog__"].sum()),
        }
        file.write(json.dumps(sp_dict))
        
    with open(os.path.join(save_path, "summary_stats.json"), "w") as file:
        file.write(json.dumps(summary_stats))


def bayeskit_save(sp, hp, draws, sampler, idx, summary_stats): 
    #  my_save(sp, hp, draws, idx)
    save_path = os.path.join(
        hp.save_dir,
        hp.posterior,
        f"{sp.sampler_type}_{idx:02d}",
        f"chain_{hp.chain:02d}",
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # save burned draws, draws, and acceptances as numpy arrays
    # np.save(os.path.join(save_path, "burned_draws"), burned_draws.astype(np.float16))
    np.savetxt(os.path.join(save_path, "draws.csv"), draws.astype(np.float64), delimiter="\t")
    
    with open(os.path.join(save_path, "params.json"), "w") as file:
        json_dict = {
            "grad_evals": sampler._model.log_density_gradient.calls,
            "proposal_nans": sampler._proposal_nans,
            "ghost_nans": sampler._ghost_nans,
        }
        
        file.write(json.dumps(json_dict))
        
    with open(os.path.join(save_path, "uturns.json"), "w") as file:
        json_dict = {
            "uturns": sampler._uturn_list
        }
        
        file.write(json.dumps(json_dict))
        
    with open(os.path.join(save_path, "summary_stats.json"), "w") as file:
        file.write(json.dumps(summary_stats))
        
    
    # accept_tuple = compute_acceptance(sampler, sp.sampler_type, burn_in, sp)
    # if accept_tuple is not None:
    #     acceptances, burned_acceptances = accept_tuple[0], accept_tuple[1]
    #     np.save(os.path.join(save_path, "burned_acceptances"), burned_acceptances)
    #     np.save(os.path.join(save_path, "acceptances"), acceptances)

    # save hyper parameters as json
    # with open(os.path.join(save_path, "hyper_params.json"), "w") as file:
    #     file.write(json.dumps(hp._asdict()))

    # save sampler parameters as json
    
        # if accept_tuple is not None:
        #     accept_list, accept_rate_total = accept_tuple[2], accept_tuple[3]
        #     sp_dict["accept_total"] = accept_rate_total
        #     for i in range(sp.num_proposals):
        #         sp_dict[f"accept_{i}"] = accept_list[i]
        
    if hp.chain == 0:
        with open(os.path.join(save_path, "..", "sampler_params.json"), "w") as file:
            sp_dict = sp._asdict()
            file.write(json.dumps(sp_dict))
