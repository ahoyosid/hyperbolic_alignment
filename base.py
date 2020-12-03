import os
import numpy as np
import pandas as pd
import seaborn as sns
from abc import abstractmethod
from matplotlib import pyplot as plt
from scipy.io import loadmat
from os.path import join as pjoin
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.init import xavier_uniform_, uniform_


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


class Atanh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        x = x.clamp(-1. + eps, 1 - eps)
        #x = torch.clamp(x, max=1. - eps)
        ctx.save_for_backward(x)
        ctx.eps = eps
        dtype = x.dtype
        x = x.double()
        res = (torch.log_(1. + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res.to(dtype)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        return g / (1 - z ** 2), None


class Asinh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        ctx.save_for_backward(x)
        ctx.eps = eps
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(eps).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        return g / (1 + z ** 2) ** 0.5, None


def atanh(x, eps=1e-15):
	x = torch.clamp(x, max=1. - eps)
	return .5 * (torch.log(1. + x) - torch.log(1. - x))

def tanh(x, eps=15):
   return torch.tanh(torch.clamp(torch.clamp(x, min=-eps), max=eps))


def acosh(x, eps=1e-15):
    return Acosh.apply(x, eps)


def asinh(x, eps=1e-15):
    return Asinh.apply(x, eps)