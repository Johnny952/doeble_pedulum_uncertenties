import torch

from .base_agent import BaseAgent

class SensitivityAgent(BaseAgent):
    def __init__(
        self, **kwargs):
        super(SensitivityAgent, self).__init__(**kwargs)
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