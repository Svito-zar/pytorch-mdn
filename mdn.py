"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math

torch.set_default_tensor_type('torch.FloatTensor')

class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi_raw = nn.Linear(in_features, num_gaussians)
        self.log_sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, minibatch):
        pi_raw = self.pi_raw(minibatch).reshape(minibatch.shape[0],-1)
        pi = nn.Softmax(dim=-1)(pi_raw)
        sigma = torch.exp(self.log_sigma(minibatch)) + 1e-10
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def log_gaussian_probability(sigma, mu, pi, target):
    """Returns the log-probability of `data` given MoG parameters `sigma` and `mu` and 'pi'.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        average of log_probabilities (BxG): Average value of the log-probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """


    target = target.unsqueeze(1).expand_as(sigma).float()

    out_shape = sigma.shape[-1]
    const = .5 * out_shape * math.log(2 * math.pi)

    exponent = torch.log(pi) - const \
               - torch.sum(((target - mu)/sigma)**2 / 2 + torch.log(sigma) ,[2])

    log_gauss = torch.logsumexp(exponent, dim=1)
    res = - torch.mean(log_gauss)
    return res


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    loss =  log_gaussian_probability(sigma, mu, pi, target)
    return loss


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample
