from .model import ActorCritic
from .aleatoric import AleatoricActorCritic
from .dropout import DropoutActorCritic
from .bnn import BNNActorCritic
from shared.models.vae import VAE

class VAEActorCritic:
    def __init__(self, **kwargs) -> None:
        self.vae = VAE(
            encoder_arc=[256, 128, 64],
            decoder_arc=[64, 128, 256],
            latent_dim=32,
            **kwargs
        )
        self.model = ActorCritic(**kwargs)

    def to(self, device):
        self.vae.to(device)
        self.model.to(device)

class BootstrapActorCritic:
    def __init__(self, nb_nets: int = 10, **kwargs) -> None:
        self.model = [AleatoricActorCritic(**kwargs) for _ in range(nb_nets)]
    
    def to(self, device):
        for model in self.model:
            model.to(device)

class Bootstrap2ActorCritic:
    def __init__(self, nb_nets: int = 10, **kwargs) -> None:
        self.model = [ActorCritic(**kwargs) for _ in range(nb_nets)]
    
    def to(self, device):
        for model in self.model:
            model.to(device)

def make_model(
        model = 'base',
        **kwargs,
    ):
    switcher = {
        'base': ActorCritic,
        'dropout': DropoutActorCritic,
        'dropout2': DropoutActorCritic,
        'bootstrap': BootstrapActorCritic,
        'bootstrap2': Bootstrap2ActorCritic,
        'sensitivity': ActorCritic,
        'bnn': BNNActorCritic,
        'aleatoric': AleatoricActorCritic,
        'vae': VAEActorCritic,
    }
    return switcher.get(model, ActorCritic)(**kwargs)
