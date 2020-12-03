import numpy as np
import torch
from sklearn.datasets import make_low_rank_matrix
from sklearn.utils import check_random_state
from scipy import linalg
from base import acosh, atanh
from manifold import Mobius


def make_domain_adaptation_toy_data(n=1000, d=2, sigma=.01, manifold=None, 
                                    bias_source=None,
                                    bias_target=None, A=None):

    manifold = Mobius(eps=1e-15) if manifold is None else manifold

    offset = torch.FloatTensor(np.array([[0, .2]]))
    
    bias_source = torch.FloatTensor(np.array([[0., .0]])) if bias_source is None else bias_source
    bias_target = torch.FloatTensor(np.array([[.4, .2]])) if bias_target is None else bias_target
    
    # Source samples
    angles = np.random.rand(n, 1) * 2 * np.pi
    xs_base = torch.FloatTensor(0.1 * np.concatenate(
        (np.sin(angles), np.cos(angles)), 
        axis=1))

    noise = torch.randn(n, 2)
    xs = manifold.add(manifold.expm_zero(sigma * noise), xs_base)
    xs[:n // 2, :] =  manifold.add(offset, xs[:n // 2, :])
    
    xs = manifold.add(bias_source, xs)
    
    # Target samples
    anglet = np.random.rand(n, 1) * 2 * np.pi
    xt_base = torch.FloatTensor(0.1 * np.concatenate(
        (np.sin(anglet), np.cos(anglet)), 
        axis=1))

    noise = torch.randn(n, 2)
    xt = manifold.add(manifold.expm_zero(sigma * noise), xt_base)
    xt[:n // 2, :] =  manifold.add(offset, xt[:n // 2, :])
    # Affine transformation
    A = torch.FloatTensor(np.array([[1.5, .7], [.7, 1.5]])) if A is None else A
    
    xt = manifold.add(bias_target, manifold.mat_mul(A, xt))

    return xs, xt, A