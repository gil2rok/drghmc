import wandb

from ..utils.posteriors import get_posterior

def configure_sampler():
    posterior_info = get_posterior(wandb.config.experiment, "~/drghmc/posteriors", "bayeskit")
    
    model, ref_draws, posterior_origin = posterior_info
    
    # get_posterior should have a clean api contract.
    # input: posteriorx
    # output: ref_draws, stan_model, data
    
    # for bk can then construct the posterior from posterior = BayesKitModel(model_path, json.dumps(data))
    
    # for stan
    
    # posterior_ingridients = (model, data, ref_draws)
    # model = CmdStanModel(stan_file=model_path)
    
    # bk
    # posterior = BayesKitModel(model, json.dumps(data))
    
    # stan
    # posterior = CmdStanModel(stan_file=model_path)
    


def generate_samples(model):
    pass


def main():
    wandb.init(group="drghmc", save_code=True)
    WANDB_JOB_TYPE = wandb.config.chain
    
    sampler = configure_sampler()
    generate_samples(sampler)

if __name__ == "__main__":
    main()