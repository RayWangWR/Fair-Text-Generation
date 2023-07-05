import numpy as np
import math

import torch
import torch.nn as nn





class CLUB(nn.Module): 

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples, w=None, scale=1):

            
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        prediction_1 = mu.unsqueeze(1)  
        y_samples_1 = y_samples.unsqueeze(0) 
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        if w is None:
            return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() * scale
        else:
             return ( (positive.sum(dim=-1) - negative.sum(dim=-1)) * w ).sum(dim=0) * scale
            
            
 

    def loglikeli(self, x_samples, y_samples, w=None):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        if w is None:
            return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)
        else:
            return ( (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1) * w ).sum(dim=0)

    def learning_loss(self, x_samples, y_samples, w=None):
        return - self.loglikeli(x_samples, y_samples, w)





class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples): 

        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


def log_sum_exp(value, dim=None, keepdim=False):

    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)




