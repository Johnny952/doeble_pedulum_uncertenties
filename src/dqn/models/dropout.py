import torch
import torch.nn as nn
from .model import Model

class Dropout(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], **kwargs):
        super(Dropout, self).__init__()
        p =  0.25
        self.model = Model(state_stack, input_dim=input_dim, output_dim=output_dim, architecture=architecture, p=p, **kwargs)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        v = self.model(x)
        return v