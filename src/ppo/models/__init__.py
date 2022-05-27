from .model import ActorCritic
from .aleatoric import AleatoricActorCritic
from .dropout import DropoutActorCritic
from .bnn import BNNActorCritic
from shared.models.vae import VAE

def make_vae(**kwargs):
    return {
        'vae': VAE(
            encoder_arc=[256, 128, 64],
            decoder_arc=[64, 128, 256],
            latent_dim=32,
            **kwargs
        ),
        'model': ActorCritic(**kwargs),
    }

def make_bootstrap(nb_nets: int = 10, **kwargs):
    return [AleatoricActorCritic(**kwargs) for _ in range(nb_nets)]

def make_bootstrap2(nb_nets: int = 10, **kwargs):
    return [ActorCritic(**kwargs) for _ in range(nb_nets)]

def make_model(
        model = 'base',
        **kwargs,
    ):
    switcher = {
        'base': ActorCritic,
        'dropout': DropoutActorCritic,
        'dropout2': DropoutActorCritic,
        'bootstrap': make_bootstrap,
        'bootstrap2': make_bootstrap2,
        'sensitivity': ActorCritic,
        'bnn': BNNActorCritic,
        'aleatoric': AleatoricActorCritic,
        # 'vae': VAETrainerModel,
    }
    return switcher.get(model, ActorCritic)(**kwargs)
