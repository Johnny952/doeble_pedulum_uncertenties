from .base_agent import BaseAgent
from .sensitivity_agent import SensitivityAgent

def make_agent(
        model,
        gamma,
        buffer,
        logger,
        device="cpu",
        max_grad_norm=0.5,
        clip_param=0.1,
        ppo_epoch=10,
        batch_size=128,   # 128
        lr=1e-3,
        nb_nets=1,
        agent='base',
    ):
    switcher = {
        'base': BaseAgent,
        # 'dropout': DropoutTrainerModel,
        # 'dropout2': DropoutTrainerModel2,
        # 'bootstrap': BootstrapTrainerModel,
        # 'bootstrap2': BootstrapTrainerModel2,
        'sensitivity': SensitivityAgent,
        # 'bnn': BNNTrainerModel,
        # 'bnn2': BNNTrainerModel2,
        # 'aleatoric': AleatoricTrainerModel,
        # 'vae': VAETrainerModel,
    }
    return switcher.get(agent, BaseAgent)(
        model,
        gamma,
        buffer,
        logger,
        device=device,
        max_grad_norm=max_grad_norm,
        clip_param=clip_param,
        ppo_epoch=ppo_epoch,
        batch_size=batch_size,
        lr=lr,
        nb_nets=nb_nets,
    )
