import torch.nn as nn
import torch
from .model import Actor
from shared.models.base import Base

class Critic(nn.Module):
    def __init__(self, state_stack, input_dim=11, architecture=[256, 128, 64]):
        super(Critic, self).__init__()

        self.base = Base(state_stack, input_dim, architecture=architecture)

        self.v = nn.Linear(architecture[-1], 1)
        self.log_var = nn.Linear(architecture[-1], 1)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        v = self.v(x)
        log_var = self.log_var(x)
        return v, log_var

class AleatoricActorCritic(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], **kwargs):
        super(AleatoricActorCritic, self).__init__()
        self.actor = Actor(state_stack, input_dim=input_dim, output_dim=output_dim, architecture=architecture)
        self.critic = Critic(state_stack, input_dim=input_dim, architecture=architecture)

    def reparameterize(self, mu, log_var):        
        sigma = torch.exp(0.5 * log_var) + 1e-5
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def forward(self, x):
        alpha, beta = self.actor(x)
        v, log_var = self.critic(x)
        reparametrization = self.reparameterize(v, log_var)
        return (alpha, beta), (reparametrization, v, log_var)
