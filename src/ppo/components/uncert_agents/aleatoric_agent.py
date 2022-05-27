import torch

from .base_agent import BaseAgent
from shared.utils.losses import det_loss

class AleatoricAgent(BaseAgent):
    def __init__(self, **kwargs):
        super(AleatoricAgent, self).__init__(**kwargs)
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
