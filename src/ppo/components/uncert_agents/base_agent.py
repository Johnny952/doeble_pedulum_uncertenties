from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger

class BaseAgent:
    def __init__(
        self,
        model: nn.Module,
        gamma,
        buffer: ReplayMemory,
        logger: Logger,
        device="cpu",
        max_grad_norm=0.5,
        clip_param=0.1,
        ppo_epoch=10,
        batch_size=128,
        lr=1e-3,
        nb_nets=None,
        **kwargs,
    ):
        self._logger = logger
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param  # epsilon in clipped loss
        self.ppo_epoch = ppo_epoch
        self._device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_nets = nb_nets

        self._buffer = buffer
        self._criterion = F.smooth_l1_loss
        self._model = model
        self.lr = lr
        if self._model is not list or self._model is not dict:
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
            logger.watch(model)
        self._nb_update = 0
        self.training_step = 0

    def select_action(self, state: np.ndarray, eval=False):
        state = torch.from_numpy(state).float().to(self._device).unsqueeze(0)

        with torch.no_grad():
            (alpha, beta), _, (epistemic, aleatoric) = self.get_uncert(state)

        if eval:
            action = alpha / (alpha + beta)
            a_logp = 0

            action = action.squeeze().cpu().numpy()
        else:
            dist = Beta(alpha, beta)
            action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1)
            action = action.squeeze().cpu().numpy()
            a_logp = a_logp.item()
        return action, a_logp, (epistemic, aleatoric)

    def chose_action(self, state: torch.Tensor):
        (alpha, beta), v = self._model(state)[:2]
        return (alpha, beta), v

    def get_uncert(self, state: torch.Tensor):
        (alpha, beta), v = self._model(state)[:2]
        epistemic = torch.mean(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
        aleatoric = torch.Tensor([0])
        return (alpha, beta), v, (epistemic, aleatoric)

    def store_transition(self, state, action, reward, next_state, a_logp):
        self._buffer.push(
            torch.flatten(torch.from_numpy(np.array(state, dtype=np.float32))).unsqueeze(dim=0),
            torch.from_numpy(np.array(action, dtype=np.float32)).unsqueeze(dim=0),
            torch.Tensor([reward]),
            torch.flatten(torch.from_numpy(np.array(next_state, dtype=np.float32))).unsqueeze(dim=0),
            torch.Tensor([a_logp]),
        )
        return self._buffer.is_memory_full()

    def empty_buffer(self):
        self._buffer.empty()

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            # TODO: Save buffer
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if eval_mode:
            self._model.eval()
        else:
            self._model.train()
        return checkpoint["epoch"]

    def unpack_buffer(self):
        dataset = self._buffer.dataset()

        states = torch.cat(dataset.state).float().to(self._device)
        actions = torch.cat(dataset.action).float().to(self._device)
        rewards = torch.cat(dataset.reward).to(self._device)
        next_states = torch.cat(dataset.next_state).float().to(self._device)
        a_logp = torch.cat(dataset.a_logp).to(self._device)

        return states, actions, rewards, next_states, a_logp

    def update(self):
        self.training_step += 1
        s, a, r, s_, old_a_logp = self.unpack_buffer()
        with torch.no_grad():
            target_v = r + self.gamma * self.chose_action(s_)[1].squeeze(dim=-1)
            adv = target_v - self.chose_action(s)[1].squeeze(dim=-1)

        for _ in range(self.ppo_epoch):
            sampler = SubsetRandomSampler(range(self._buffer._capacity))
            losses = self.train_once(
                self._model,
                self._optimizer,
                target_v,
                adv,
                old_a_logp,
                s,
                a,
                sampler,
            )

            self._logger.log(losses)
            self._nb_update += 1

    def train_once(
        self, net, optimizer, target_v, adv, old_a_logp, s, a, rand_sampler
    ):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)
        losses = {
            'Action Loss': 0,
            'Value Loss': 0,
            'Total Loss': 0,
            "Update Step": self._nb_update,
        }
        
        for index in sampler:
            prediction = net(s[index])
            alpha, beta = prediction[0][0].squeeze(dim=-1), prediction[0][1].squeeze(dim=-1)
            
            dist = Beta(alpha, beta)
            a_logp = dist.log_prob(a[index])

            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.get_value_loss(prediction, target_v[index])
            loss = action_loss + 2.0 * value_loss

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            optimizer.step()

            losses["Action Loss"] += action_loss.item()
            losses["Value Loss"] += value_loss.item()
            losses["Total Loss"] += loss.item()
        return losses

    def get_value_loss(self, prediction, target_v):
        return self._criterion(prediction[1].squeeze(dim=-1), target_v)