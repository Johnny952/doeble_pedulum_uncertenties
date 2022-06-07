import torch
from torch import optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .base_agent import BaseAgent


class VAEAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._optimizer = optim.Adam(self._model.model.parameters(), lr=self.lr)
        self._vae_optimizer = optim.Adam(self._model.vae.parameters(), lr=self.lr)

        self._nb_vae_epochs = 10
        self._kld_scale = 0.015

        self._nb_vae_update = 0

    def chose_action(self, state: torch.Tensor):
        (alpha, beta), v = self._model.model(state)[:2]
        return (alpha, beta), v

    def get_uncert(self, state: torch.Tensor):
        (alpha, beta), v = self._model.model(state)[:2]
        [_, log_var] = self._model.vae.encode(state)
        epistemic = torch.sum(log_var)
        aleatoric = torch.Tensor([0])
        return (alpha, beta), v, (epistemic, aleatoric)

    def save(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {
            'epoch': epoch,
            'model_state_dict': self._model.model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'vae_state_dict': self._model.vae.state_dict(),
            'vae_optimizer_state_dict': self._vae_optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._model.vae.load_state_dict(checkpoint['vae_state_dict'])
        self._vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        if eval_mode:
            self._model.model.eval()
            self._model.vae.eval()
        else:
            self._model.model.train()
            self._model.vae.train()
        return checkpoint['epoch']

    def train_once_vae(self, s, rand_sampler):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)
        losses = {
            'KLD Loss': 0,
            'Encode Loss': 0,
            'VAE Loss': 0,
            'VAE Step': self._nb_vae_update,
        }
        for index in sampler:
            [decoding, input, mu, log_var] = self._vae(s[index])
            l = self._model.vae.loss_function(decoding, input, mu, log_var, M_N=self._kld_scale)
            loss = l['loss']
            recons_loss = l['Reconstruction_Loss']
            kld_loss = l['kld_loss']
            self._vae_optimizer.zero_grad()
            loss.backward()
            self._vae_optimizer.step()

            losses['KLD Loss'] += kld_loss.item()
            losses['Encode Loss'] += recons_loss.item()
            losses['VAE Loss'] += loss.item()
        
        return losses

    def update(self):
        self.training_step += 1
        s, a, r, s_, old_a_logp = self.unpack_buffer()
        with torch.no_grad():
            target_v = r + self.gamma * self.chose_action(s_)[1].squeeze(dim=-1)
            adv = target_v - self.chose_action(s)[1].squeeze(dim=-1)

        self._model.model.train()
        self._model.vae.eval()

        for _ in range(self.ppo_epoch):
            sampler = SubsetRandomSampler(range(self._buffer._capacity))
            losses = self.train_once(
                self._model.model,
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

        self._model.model.eval()
        self._model.vae.train()

        for _ in range(self._nb_vae_epochs):
            sampler = SubsetRandomSampler(range(self.buffer_capacity))
            losses = self.train_once_vae(s, sampler)
            self._logger.log(losses)
            self._nb_vae_update += 1