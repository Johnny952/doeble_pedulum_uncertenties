import torch.nn as nn
from shared.models.base import Base

class Model(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, state_stack, input_dim=11, output_dim=1, mid_dim=32, architecture=[256, 128, 64], **kwargs):
        super(Model, self).__init__()

        self.base = Base(state_stack, input_dim, architecture=architecture)

        self.v = nn.Sequential(
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
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

class Actor(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64]):
        super(Actor, self).__init__()

        self.base = Base(state_stack, input_dim, architecture=architecture)

        self.alpha_head = nn.Sequential(
            nn.Linear(architecture[-1], output_dim),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(architecture[-1], output_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return alpha, beta

class Critic(nn.Module):
    def __init__(self, state_stack, input_dim=11, architecture=[256, 128, 64]):
        super(Actor, self).__init__()

        self.base = Base(state_stack, input_dim, architecture=architecture)

        self.v = nn.Linear(architecture[-1], 1)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        v = self.v(x)
        return v

class ActorCritic(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, mid_dim=32, architecture=[256, 128, 64], **kwargs):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_stack, input_dim=input_dim, output_dim=output_dim, architecture=architecture)
        self.critic = Critic(state_stack, input_dim=input_dim, architecture=architecture)

    def forward(self, x):
        alpha, beta = self.actor(x)
        v = self.critic(x)
        return (alpha, beta), v