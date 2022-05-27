import torch

from .base_agent import BaseAgent

class DropoutAgent2(BaseAgent):
    def __init__(self, **kwargs):
        super(DropoutAgent2, self).__init__(**kwargs)

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

        epistemic = torch.var(alpha_list / (alpha_list + beta_list)) + torch.var(beta_list)
        aleatoric = torch.tensor([0])

        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)

    def load(self, path, eval_mode=False):
        return super(DropoutAgent2, self).load(path, eval_mode=False)