import torch.nn as nn
import torchbnn as bnn
from shared.models.bnn import BNNBase

class Actor(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64]):
        super(Actor, self).__init__()

        self.base = BNNBase(state_stack, input_dim, architecture=architecture)

        self.alpha_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=architecture[-1], out_features=output_dim),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=architecture[-1], out_features=output_dim),
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
        super(Critic, self).__init__()

        self.base = BNNBase(state_stack, input_dim, architecture=architecture)

        self.v = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=architecture[-1], out_features=1)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        v = self.v(x)
        return v

class BNNActorCritic(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], **kwargs):
        super(BNNActorCritic, self).__init__()
        self.actor = Actor(state_stack, input_dim=input_dim, output_dim=output_dim, architecture=architecture)
        self.critic = Critic(state_stack, input_dim=input_dim, architecture=architecture)

    def forward(self, x):
        alpha, beta = self.actor(x)
        v = self.critic(x)
        return (alpha, beta), v