import torch.nn as nn
import torch

from .base_agent import BaseAgent
from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger
from shared.utils.losses import det_loss

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
        self._criterion = det_loss
        self._weight_decay = 1e-10

    def chose_action(self, state: torch.Tensor):
        (alpha, beta), (_, v, _) = self._model(state)
        return (alpha, beta), v

    def get_uncert(self, state: torch.Tensor):
        (alpha, beta), (_, mu, log_var) = self._model(state)
        epistemic = torch.tensor([0])
        aleatoric = log_var
        return (alpha, beta), mu, (epistemic, aleatoric)

    def get_value_loss(self, prediction, target_v):
        _, (v, mu, log_var) = prediction
        return self._criterion(v.squeeze(dim=-1), target_v, mu, log_var, weight_decay=self._weight_decay)
