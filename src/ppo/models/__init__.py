from .model import Model, ActorCritic
from .aleatoric import Aleatoric

def make_bootstrap(
        state_stack: int,
        input_dim: int=11,
        output_dim: int=1,
        architecture: "list[int]"=[256, 128, 64],
        mid_dim: int=32,
        nb_nets: int = 10,
    ):
    return [Aleatoric(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
        mid_dim=mid_dim,
    ) for _ in range(nb_nets)]

def make_bootstrap2(
        state_stack: int,
        input_dim: int=11,
        output_dim: int=1,
        architecture: "list[int]"=[256, 128, 64],
        mid_dim: int=32,
        nb_nets: int = 10,
    ):
    return [Model(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
        mid_dim=mid_dim,
    ) for _ in range(nb_nets)]

def make_model(
        state_stack: int,
        input_dim: int = 11,
        output_dim: int = 1,
        architecture: "list[int]" = [256, 128, 64],
        model = 'base',
        mid_dim: int = 32,
        nb_nets: int = 10,
    ):
    switcher = {
        'base': ActorCritic,
        # 'dropout': DropoutTrainerModel,
        # 'dropout2': DropoutTrainerModel2,
        'bootstrap': make_bootstrap,
        'bootstrap2': make_bootstrap2,
        'sensitivity': Model,
        # 'bnn': BNNTrainerModel,
        # 'bnn2': BNNTrainerModel2,
        'aleatoric': Aleatoric,
        # 'vae': VAETrainerModel,
    }
    return switcher.get(model, ActorCritic)(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
        mid_dim=mid_dim,
        nb_nets=nb_nets,
    )
