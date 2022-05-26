from .model import ActorCritic
from .aleatoric import AleatoricActorCritic
from .dropout import DropoutActorCritic
from .bnn import BNNActorCritic

def make_bootstrap(
        state_stack: int,
        input_dim: int=11,
        output_dim: int=1,
        architecture: "list[int]"=[256, 128, 64],
        nb_nets: int = 10,
    ):
    return [AleatoricActorCritic(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
    ) for _ in range(nb_nets)]

def make_bootstrap2(
        state_stack: int,
        input_dim: int=11,
        output_dim: int=1,
        architecture: "list[int]"=[256, 128, 64],
        nb_nets: int = 10,
    ):
    return [ActorCritic(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
    ) for _ in range(nb_nets)]

def make_model(
        state_stack: int,
        input_dim: int = 11,
        output_dim: int = 1,
        architecture: "list[int]" = [256, 128, 64],
        model = 'base',
        nb_nets: int = 10,
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
    return switcher.get(model, ActorCritic)(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
        nb_nets=nb_nets,
    )
