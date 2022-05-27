import torch
import torchbnn as bnn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
from .base_agent import BaseAgent

class BNNAgent(BaseAgent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self._kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.sample_nbr = 50
        self.complexity_cost_weight = 1e-6

    def chose_action(self, state: torch.Tensor):
        alpha_list = []
        beta_list = []
        v_list = []
        for _ in range(self.nb_nets):
            (alpha, beta), v = self._model(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0)

    def get_uncert(self, state: torch.Tensor):
        alpha_list = []
        beta_list = []
        v_list = []
        for _ in range(self.nb_nets):
            (alpha, beta), v = self._model(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)

        epistemic = torch.mean(
            torch.var(alpha_list / (alpha_list + beta_list), dim=0))
        aleatoric = torch.Tensor([0])

        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)

    def update(self):
        self.training_step += 1
        s, a, r, s_, old_a_logp = self.unpack_buffer()
        with torch.no_grad():
            target_v = r + self.gamma * self.chose_action(s_)[1].squeeze(dim=-1)
            adv = target_v - self.chose_action(s)[1].squeeze(dim=-1)

        for _ in range(self.ppo_epoch):
            rand_sampler = SubsetRandomSampler(range(self._buffer._capacity))
            sampler = BatchSampler(rand_sampler, self.batch_size, False)
            losses = {
                'Action Loss': 0,
                'Value Loss': 0,
                'Total Loss': 0,
                "Update Step": self._nb_update,
            }
            for index in sampler:
                loss = 0
                for _ in range(self.sample_nbr):
                    prediction = self._model(s[index])
                    alpha, beta = prediction[0][0].squeeze(dim=-1), prediction[0][1].squeeze(dim=-1)
                    
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index])
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self.get_value_loss(prediction, target_v[index])
                    kl_loss = self._kl_loss(self._model) * self.complexity_cost_weight

                    loss += action_loss + 2. * value_loss + kl_loss

                    losses["Action Loss"] += action_loss.item()
                    losses["Value Loss"] += value_loss.item()
                    losses["Total Loss"] += loss.item()
                self._optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                self._optimizer.step()

            self._logger.log(losses)
            self._nb_update += 1
