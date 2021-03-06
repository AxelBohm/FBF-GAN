#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.
# modifications by Axel Boehm (axel.boehm@univie.ac.at) and
# Michael Sedlmayer (michael.sedlmayer@univie.ac.at).

import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torch import where, add, abs, zeros_like, ones_like
import torch.nn.functional as F


def clip_weights(params, clip=0.01):
    for p in params:
        p.clamp_(-clip, clip)


def unormalize(x):
    return x/2. + 0.5


def sample(name, size):
    if name == 'normal':
        return torch.zeros(size).normal_()
    elif name == 'uniform':
        return torch.zeros(size).uniform_()
    else:
        raise ValueError()


def weight_init(m, mode='normal'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        if mode == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'kaimingu':
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, 0.8)


def compute_gan_loss(p_true, p_gen, mode='gan', gen_flag=False):
    if mode == 'ns-gan' and gen_flag:
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - \
            (p_gen.clamp(max=0) - torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'gan' or mode == 'gan++':
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - \
            (p_gen.clamp(min=0) + torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'wgan':
        loss = p_true.mean() - p_gen.mean()
    else:
        raise NotImplementedError()

    return loss


def prox_1norm(data, lam):
    """compute the proximal operator of ||.||_1 with stepsize $lam$
    """
    data = where(abs(data) <= lam, zeros_like(data), data)
    data = where(data > lam, add(data, -lam, ones_like(data)), data)
    data = where(data < -lam, add(data, lam, ones_like(data)), data)
    # alternative to compute prox via Moreau decomposition
    # p.data = torch.add(p.data, -l, torch.clamp(torch.mul(p.data, 1/l), -1, 1))
    return data


def spectral_normalize(W, u, iter=1):

    sigma, u = max_singular_value(W.reshape(W.shape[0], -1), u, iter)

    return W/sigma, u


def max_singular_value(W, u, iter):
    """ computes largest singular value of rectangular tensor
    """

    if u is None:
        u = torch.randn(size=(1, W.shape[0])).cuda(0)

    for _ in range(iter):
        v = F.normalize(torch.matmul(u, W))
        u = F.normalize(torch.matmul(v, W.t()))
    sigma = torch.matmul(u, torch.matmul(W, v.t()))[0][0]

    return sigma, u
