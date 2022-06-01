import torch.nn as nn
import torch
from shared.models.base import Base

class Aleatoric(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], **kwargs):
        super(Aleatoric, self).__init__()
        self.base = Base(state_stack, input_dim, architecture=architecture)
        self.v = nn.Sequential(
            nn.Linear(architecture[-1], output_dim),
            nn.Softplus()
        )
        self.log_var = nn.Sequential(
            nn.Linear(architecture[-1], output_dim),
            nn.Softplus()
        )

    def reparameterize(self, mu, log_var):    
        sigma = torch.exp(0.5 * log_var) + 1e-5
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        mu = self.v(x)
        log_var = self.log_var(x)
        reparametrization = self.reparameterize(mu, log_var)
        return reparametrization, mu, log_var
