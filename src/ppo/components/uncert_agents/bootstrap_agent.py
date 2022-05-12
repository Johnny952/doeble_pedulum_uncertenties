import torch.nn as nn
import torch
import torch.optim as optim

from .base_agent import BaseAgent
from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger
from shared.utils.losses import ll_gaussian
from shared.utils.mixtureDist import GaussianMixture

class BootstrapAgent(BaseAgent):
    def __init__(
        self,
        model: "list[nn.Module]",
        gamma, buffer: ReplayMemory,
        logger: Logger,
        device="cpu",
        max_grad_norm=0.5,
        clip_param=0.1,
        ppo_epoch=10,
        batch_size=128,
        lr=0.001,
        nb_nets=None
    ):
        super().__init__(model, gamma, buffer, logger, device, max_grad_norm, clip_param, ppo_epoch, batch_size, lr, nb_nets)

        self._criterion = ll_gaussian
        self._value_scale = 1 / nb_nets
        self._optimizer = [optim.Adam(net.parameters(), lr=lr) for net in self._model]

    def chose_action(self, state: torch.Tensor):
        alpha_list = []
        beta_list = []
        log_sigma_list = []
        v_list = []
        for net in self._model:
            (alpha, beta), v, sigma = net(state)
            log_sigma_list.append(sigma)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        sigma_list = torch.exp(torch.stack(log_sigma_list))
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)
        distribution = GaussianMixture(v_list.squeeze(
            dim=-1), sigma_list.squeeze(dim=-1), device=self.device)
        v = distribution.mean.unsqueeze(dim=-1)
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), v

    def get_uncert(self, state: torch.Tensor):
        alpha_list = []
        beta_list = []
        log_sigma_list = []
        v_list = []
        for net in self._model:
            (alpha, beta), v, sigma = net(state)
            log_sigma_list.append(sigma)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        sigma_list = torch.exp(torch.stack(log_sigma_list))
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)

        distribution = GaussianMixture(v_list.squeeze(
            dim=1), sigma_list.squeeze(dim=1), device=self.device)
        epistemic = distribution.std
        aleatoric = torch.tensor([0])
        v = distribution.mean
        # v = torch.mean(v_list, dim=0)
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), v, (epistemic, aleatoric)

    def update(self):
        self.training_step += 1
        s, a, r, s_, old_a_logp = self.unpack_buffer()
        with torch.no_grad():
            target_v = r + self.gamma * self.chose_action(s_)[1].squeeze(dim=-1)
            adv = target_v - self.chose_action(s)[1].squeeze(dim=-1)

        # Random bagging
        # indices = [torch.utils.data.RandomSampler(range(
        #     self.buffer_capacity), num_samples=self.buffer_capacity, replacement=True) for _ in range(self.nb_nets)]
        # Random permutation
        indices = [torch.randperm(self._buffer._capacity) for _ in range(self.nb_nets)]

        for _ in range(self.ppo_epoch):

            for net, optimizer, index in zip(self._model, self._optimizer, indices):
                losses = self.train_once(
                    net,
                    optimizer,
                    target_v, adv, old_a_logp, s, a, index)

            self._logger.log(losses)
            self._nb_update += 1
    
    def get_value_loss(self, prediction, target_v):
        v = prediction[1]
        sigma = prediction[-1]
        return self._criterion(v, target_v, sigma) * self._value_scale

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {'epoch': epoch}
        for idx, (net, optimizer) in enumerate(zip(self._model, self._optimizer)):
            tosave['model_state_dict{}'.format(idx)] = net.state_dict()
            tosave['optimizer_state_dict{}'.format(
                idx)] = optimizer.state_dict()
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        for idx in range(len(self._model)):
            self._model[idx].load_state_dict(
                checkpoint['model_state_dict{}'.format(idx)])
            self._optimizer[idx].load_state_dict(
                checkpoint['optimizer_state_dict{}'.format(idx)])
            if eval_mode:
                self._model[idx].eval()
            else:
                self._model[idx].train()
        return checkpoint['epoch']