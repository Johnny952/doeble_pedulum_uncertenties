import torch.nn as nn
import torch

from .base_agent import BaseAgent
from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger

class AleatoricAgent(BaseAgent):
    def __init__(
        self,
        model: nn.Module,
        gamma,
        buffer: ReplayMemory,
        logger: Logger,
        device="cpu",
        max_grad_norm=0.5,
        clip_param=0.1,
        ppo_epoch=10,
        batch_size=128,
        lr=1e-3,
        nb_nets=None,
    ):
        super(AleatoricAgent, self).__init__(
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