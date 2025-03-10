# ---------------------------------------------------------------
# This file contains a modifed version of 
# code taken from https://github.com/NVlabs/NVAE
#-----------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import one_hot


class Normal:
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = mu
        self.mu = 5. * torch.tanh(mu / 5.)            # soft differentiable clamp between [-5, 5]
        self.log_sigma = 5. * torch.tanh(log_sigma / 5.)   # soft differentiable clamp between [-5, 5]
        self.sigma = torch.exp(self.log_sigma) + 1e-2      # we don't need this after soft clamp
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        eps = torch.Tensor(self.mu.size()).cuda().normal_()
        return eps * self.sigma + self.mu, eps

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    
    def inverse_reparam(self, samples, mu, log_sigma): #shifts and scales samples from one distribution, with params of another distribution
        return (samples - mu) / (torch.exp(log_sigma) + 1e-2)

class NormalDecoder:
    def __init__(self, param, num_bits=8):
        B, C, H, W = param.size()
        self.num_c = C // 2
        mu = param[:, :self.num_c, :, :]                                 # B, 3, H, W
        log_sigma = param[:, self.num_c:, :, :]                          # B, 3, H, W
        self.dist = Normal(mu, log_sigma)
        m = torch.clamp(mu, -1, 1.)
        self.mean = m/2. + 0.5        

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0
        return self.dist.log_p(samples)

    def sample(self, t=1.):
        x, _ = self.dist.sample()
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x

class DiscLogistic:
    def __init__(self, param):
        B, C, H, W = param.size()
        self.num_c = C // 2
        self.means = param[:, :self.num_c, :, :]                              # B, 3, H, W
        self.log_scales = torch.clamp(param[:, self.num_c:, :, :], min=-8.0)  # B, 3, H, W

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == self.num_c

        centered = samples - self.means                                         # B, 3, H, W
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(127.5))
        # woow the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, H, W

        return log_probs

    def sample(self):
        u = torch.Tensor(self.means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
        x = self.means + torch.exp(self.log_scales) * (torch.log(u) - torch.log(1. - u))            # B, 3, H, W
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x


class DiscMixLogistic:
    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)                    # B, 3, 3 * M, H, W
        self.means = l[:, :, :num_mix, :, :]                                          # B, 3, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)   # B, 3, M, H, W
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])              # B, 3, M, H, W
        self.max_val = 2. ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == 3, 'only RGB images are considered.'

        samples = samples.unsqueeze(4)                                                  # B, 3, H , W
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 3, M, H, W
        mean1 = self.means[:, 0, :, :, :]                                               # B, M, H, W
        mean2 = self.means[:, 1, :, :, :] + \
                self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]                     # B, M, H, W
        mean3 = self.means[:, 2, :, :, :] + \
                self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
                self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]                     # B, M, H, W

        mean1 = mean1.unsqueeze(1)                          # B, 1, M, H, W
        mean2 = mean2.unsqueeze(1)                          # B, 1, M, H, W
        mean3 = mean3.unsqueeze(1)                          # B, 1, M, H, W
        means = torch.cat([mean1, mean2, mean3], dim=1)     # B, 3, M, H, W
        centered = samples - means                          # B, 3, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, M, H, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W
        return torch.logsumexp(log_probs, dim=1)                                      # B, H, W

    def sample(self, t=1.):
        gumbel = -torch.log(- torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x
    
    def mean(self):
        sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 3, H, W

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means                                                                              # B, 3, H, W
        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x
    


if __name__ == '__main__':
    """
    mu1 = 0.5 * torch.ones(size=(100000, 1)).cuda()
    sigma1 = 0.1 * torch.ones(size=(100000, 1)).cuda() - 1e-2
    dist1 = Normal(mu1, sigma1)
    s1 = dist1.cells()

    mn = torch.mean(s1)
    std = torch.mean(s1 * s1) - mn * mn
    print(mn, std)

    mu2 = 2.5 * torch.ones(size=(100000, 1)).cuda()
    sigma2 = 0.5 * torch.ones(size=(100000, 1)).cuda() - 1e-2
    dist2 = Normal(mu2, sigma2)

    kl = torch.mean(dist1.log_p(s1) - dist2.log_p(s1))
    print(kl)
    print(dist1.kl(dist2))
    """

    # Test DiscMixLogistic
    """
    num_mix = 2
    param = torch.from_numpy(np.random.randn(1, num_mix * 10, 3, 4)).float().cuda() / 3.

    num_sample = 100000
    param = param.expand(num_sample, -1, -1, -1)

    dist = DiscMixLogistic(param, num_mix)

    cells = dist.cells()
    log_p = dist.log_p(cells)
    print(torch.mean(log_p, dim=0))
    """
    # Test DiscMixLogistic
    mean = torch.from_numpy(0.3 * np.ones((1, 1, 1, 1))).float().cuda()
    log_scale = torch.from_numpy(-10 * np.ones((1, 1, 1, 1))).float().cuda()
    param = torch.cat([mean, log_scale], dim=1)
    num_sample = 100000
    param = param.expand(num_sample, -1, -1, -1)
    dist = DiscLogistic(param)
    sample = dist.sample()
    log_p = dist.log_prob(sample)

    print(torch.mean(sample, dim=0))
    print(torch.std(sample, dim=0))
    print(torch.mean(log_p, dim=0))

    param = torch.cat([mean, log_scale], dim=1)
    num_sample = 256
    param = param.expand(num_sample, -1, -1, -1)
    dist = DiscLogistic(param)
    x = torch.from_numpy(np.arange(0., 1.0, step=1./256).reshape(-1, 1, 1, 1)).float().cuda()
    log_p = dist.log_prob(x)
    log_p = torch.flatten(log_p).cpu().numpy()
    print(x.size())

    print(torch.logsumexp(dist.log_prob(x).flatten(), dim=0))

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.plot(np.exp(log_p))
    plt.show()

