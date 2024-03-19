from omegaconf import DictConfig
from omegaconf.resolvers import Resolver
from hydra.core.plugins import Plugins


class CustomResolver(Resolver):
    def resolve(self, key: str, config: DictConfig) -> any:
        if key == "concat_sampler_params":
            params = config.sampler.params
            if isinstance(params, DictConfig):
                return "__".join(
                    [f"{k}={v}" for k, v in params.items() and v is not None]
                )
            else:
                return None
        else:
            return None

def custom_resolver(config_loader, config_name):
    return CustomResolver()

# Register the resolver with Hydra
Plugins.instance().config_resolver.register("custom_resolver", custom_resolver)
