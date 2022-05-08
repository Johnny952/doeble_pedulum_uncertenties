import torch.nn as nn
import torch

from .base_agent import BaseAgent
from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger

class SensitivityAgent(BaseAgent):
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
        super(SensitivityAgent, self).__init__(
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
        self._noise_variance = 0.1

    def get_uncert(self, state: torch.Tensor):
        size = (self.nb_nets, state.shape[1])
        rand_dir = torch.normal(
            torch.zeros(size), self._noise_variance*torch.ones(size)
        ).double().to(self.device)
        rand_dir += state
        #rand_dir[rand_dir > self.input_range[1]] = self.input_range[1]
        #rand_dir[rand_dir < self.input_range[0]] = self.input_range[0]

        # Estimate uncertainties
        (alpha, beta), v = self._model(rand_dir)

        epistemic = torch.mean(torch.var(alpha / (alpha + beta), dim=0))
        aleatoric = torch.Tensor([0])

        # Predict
        (alpha, beta), v = self._model(state)
        return (alpha, beta), v, (epistemic, aleatoric)