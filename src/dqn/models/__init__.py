from .model import Model
from .aleatoric import Aleatoric
from .dropout import Dropout
from .bnn import BNN
from shared.models.vae import VAE

class VAEModel:
    def __init__(self, **kwargs) -> None:
        self.vae = VAE(
            encoder_arc=[256, 128, 64],
            decoder_arc=[64, 128, 256],
            latent_dim=32,
            **kwargs
        )
        self.model = Model(**kwargs)

    def to(self, device):
        self.vae.to(device)
        self.model.to(device)

class Bootstrap:
    def __init__(self, nb_nets: int = 10, **kwargs) -> None:
        self.model = [Aleatoric(**kwargs) for _ in range(nb_nets)]
    
    def to(self, device):
        for model in self.model:
            model.to(device)

class Bootstrap2:
    def __init__(self, nb_nets: int = 10, **kwargs) -> None:
        self.model = [Model(**kwargs) for _ in range(nb_nets)]
    
    def to(self, device):
        for model in self.model:
            model.to(device)

def make_model(
        model = 'base',
        **kwargs,
    ):
    switcher = {
        'base': Model,
        'dropout': Dropout,
        'dropout2': Dropout,
        'bootstrap': Bootstrap,
        'bootstrap2': Bootstrap2,
        'sensitivity': Model,
        'bnn': BNN,
        'aleatoric': Aleatoric,
        'vae': VAEModel,
    }
    return switcher.get(model, Model)(**kwargs)
