import torch.nn as nn
import torch
from shared.models.base import Base

class Aleatoric(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, state_stack, input_dim=11, output_dim=1, mid_dim=32, architecture=[256, 128, 64], **kwargs):
        super(Aleatoric, self).__init__()

        self.base = Base(state_stack, input_dim, architecture=architecture)

        self.v = nn.Sequential(
            nn.Linear(architecture[-1], mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )
        self.log_var = nn.Sequential(
            nn.Linear(architecture[-1], mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )
        self.fc = nn.Sequential(nn.Linear(architecture[-1], mid_dim), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(mid_dim, output_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(mid_dim, output_dim), nn.Softplus())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        v = self.v(x)
        log_var = self.log_var(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        reparametrization = self.reparameterize(v, log_var)

        return (alpha, beta), (reparametrization, v, log_var)

    def reparameterize(self, mu, log_var):        
        sigma = torch.exp(0.5 * log_var) + 1e-5
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon