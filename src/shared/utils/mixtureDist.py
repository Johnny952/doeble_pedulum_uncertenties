import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

class Mixture:
    def __init__(self, means, dev=1, device='cpu'):
        weights = torch.ones(means.shape[0]).to(device)
        self.dims = list(means.shape)[1:]
        means_ = means.reshape((means.shape[0], -1)).to(device).double()
        stdevs = dev*torch.ones((means_.shape[0], 1)).to(device)
        self.device = device

        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means_, stdevs), 1)
        self.gmm = MixtureSameFamily(mix, comp)
    
    def sample(self, n_samples: int):
        return self.gmm.sample(sample_shape=(n_samples,)).reshape(tuple([n_samples]+self.dims)).float().to(self.device)
    
    def logp(self, x):
        return self.gmm.log_prob(x.reshape((x.shape[0], -1))).to(self.device)


class GaussianMixture:
    def __init__(self, means, stdevs, device='cpu'):
        weights = torch.ones(means.shape[0]).to(device)
        self.dims = list(means.shape)[1:]
        self.device = device

        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means, stdevs), 1)
        self.gmm = MixtureSameFamily(mix, comp)

        self.var = self.gmm.variance
        self.std = self.gmm.stddev
        self.mean = self.gmm.mean
    
    def sample(self, n_samples: int):
        return self.gmm.sample(sample_shape=(n_samples,)).reshape(tuple([n_samples]+self.dims)).float().to(self.device)
    
    def logp(self, x):
        return self.gmm.log_prob(x.reshape((x.shape[0], -1))).to(self.device)

if __name__ == "__main__":
    # 10 Image 50x50
    dims = (10, 50, 50)
    # dims = (10, 1)
    means = torch.arange(0, dims[0]*dims[1]*dims[2], 1).view(dims)
    # means = torch.arange(0, dims[0]*dims[1], 1).view(dims)

    gen = Mixture(means, dev=0.2)
    x = torch.arange(-1, 11, 0.01)
    random_x = gen.sample(200)

    # plt.plot(x, np.exp(gen.logp(x)))
    # plt.plot(random_x, np.exp(gen.logp(random_x)), '*')
    # plt.show()