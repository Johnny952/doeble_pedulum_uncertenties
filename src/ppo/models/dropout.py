import torch.nn as nn
from .model import Actor, Critic

class DropoutActorCritic(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], **kwargs):
        super(DropoutActorCritic, self).__init__()
        actor_p =  0.25
        critic_p = 0.25
        self.actor = Actor(state_stack, input_dim=input_dim, output_dim=output_dim, architecture=architecture, p=actor_p)
        self.critic = Critic(state_stack, input_dim=input_dim, architecture=architecture, p=critic_p)

    def forward(self, x):
        alpha, beta = self.actor(x)
        v = self.critic(x)
        return (alpha, beta), v