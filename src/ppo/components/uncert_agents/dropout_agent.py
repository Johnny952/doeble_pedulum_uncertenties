import torch
import torch.optim as optim

from .base_agent import BaseAgent

class DropoutAgent(BaseAgent):
    def __init__(self, **kwargs):
        super(DropoutAgent, self).__init__(**kwargs)
        self.lengthscale = 0.01
        self.weight_decay = 1e-8
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_uncert(self, state):
        # Estimate uncertainties
        alpha_list = []
        beta_list = []
        v_list = []
        for _ in range(self.nb_nets):
            with torch.no_grad():
                (alpha, beta), v = self._model(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)

        tau = self.lengthscale * (1. - self.prob) / \
            (2. * state.shape[0] * self.weight_decay)
        epistemic = torch.var(alpha_list / (alpha_list + beta_list)) + 1. / tau
        aleatoric = torch.tensor([0])

        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)

    def load(self, path, eval_mode=False):
        return super(DropoutAgent, self).load(path, eval_mode=False)