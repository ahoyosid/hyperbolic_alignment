import numpy as np
import torch
from abc import abstractmethod
from base import atanh, tanh, asinh


class Manifold(object):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def dim(dim):
        return dim

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError

    @abstractmethod
    def rgrad(self, p, d_p):
        """
        Riemannian gradient
        """
        raise NotImplementedError

class Euclidean(Manifold):

    def add(self, x, y):
        return x + y

    def expm(self, x, v):
        return x + v
    
    def logm(self, x, y):
        return y - x

    def logm_zero(self, x):
        return x

    def expm_zero(self, x):
        return x
    
    def scalar_mul(self, r, x):
        return r * x

    def mat_mul(self, M, x):
        return x.mm(M.t())
    
    def distance(self, x, y, dim=-1, keepdim=True):
        return (x - y).norm(dim=dim, keepdim=keepdim)
    
    def barycenter_mapping(self, x, coupling, n=None, kind="uniform", dim=-1, 
                           keepdim=True):

        if (kind == "uniform") and (n is not None):
            return n * coupling.mm(x)
        else:
            return coupling.mm(x) / coupling.sum(dim=dim, keepdim=keepdim)

    def geodesic(self, x, y, t):
        """
            x \oplus (-x \oplus y) \otimes t
        """
        return self.add(x, self.scalar_mul(t, self.add(-x, y)))   

    def mean(self, x, dim=0, keepdim=True):
        return x.mean(dim=dim, keepdim=keepdim)         
    
    def rgrad(self, p, d_p):
        return d_p

        
class HyperbolicManifold(Manifold):
    """
        s: Radius of the disk
    """
    def loid2ball(self, u):
        """ hyperboloid model
            [x_1 / (1 + x_0), ... x_n / (1 + x_0)] in B^n
        """
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)    
    
    def ball2loid(self, x):
        y = x.clone()   
        norm_y = y.norm(2, dim=-1, keepdim=True).pow(2)
        y0 = 2. / (1 - norm_y) - 1
        return torch.cat([y0, y * (y0 + 1.)], dim=-1)


class HyperbolicDisk(Manifold):

    def __init__(self, s=1, eps=1e-15):
        self.s = s
        self.eps = eps
        self.boundary = self.s - eps
        self.max_norm = self.boundary 
        self.min_norm = eps  

    @abstractmethod
    def add(self, x, y):
        """
        Addition
        """
        raise NotImplementedError        

    def klein2poincare(self, x, dim=-1, keepdim=True):
        s_sqr = self.s ** 2 
        norm_x_sqr = x.norm(dim=dim, keepdim=keepdim).pow(2) / s_sqr
        norm_x_sqr.data.clamp_(min=self.min_norm)
        den = 1 + (1 - norm_x_sqr).pow(.5)
        return (x / den)

    def poincare2klein(self, x, dim=-1, keepdim=True):
        s_sqr = self.s ** 2 
        norm_x_sqr = x.norm(dim=dim, keepdim=keepdim).pow(2) / s_sqr
        norm_x_sqr.data.clamp_(min=self.min_norm)
        return (2 * x) / (1 + norm_x_sqr)

    def gamma(self, x, dim=-1, keepdim=True):
        """ Lorentz gamma factor in the s-ball
        
        Eq. (3.131) in [1].
            
        Analytic hyperbolic geometry and Albert Einstein special theory of relativity.
        """
        s_sqr = self.s ** 2
        norm_x_sqr = x.norm(dim=dim, keepdim=keepdim).pow(2)
        norm_x_sqr.data.clamp_(min=self.min_norm)
        scaled_norm_x = norm_x_sqr / s_sqr
        return 1 / (1. - scaled_norm_x).pow(.5)

    def proj2ball(self, x, proj_eps=1e-5, dim=-1, keepdim=True):
        x_norm = x.norm(p=2, dim=dim, keepdim=keepdim)
        mask = x_norm >= self.s
        scale = torch.ones(x_norm.size())
        scale[mask] = (self.s - proj_eps) / x_norm[mask]
        return x * scale

    def scalar_mul(self, r, x, dim=-1, keepdim=True):
        x = x + self.eps
        s = self.s
        norm_x = x.norm(2, dim=dim, keepdim=keepdim)
        norm_x.data.clamp_(min=self.min_norm)
        t =  r * atanh((norm_x / s), self.eps)
        t.data.clamp_(min=self.eps)
        return self.proj2ball(s * tanh(t) * x.div(norm_x))

    def _distance(self, x, y):
        return atanh(self.gyrodistance(x, y) / self.s)

    def gyrodistance(self, x, y, dim=-1, keepdim=True):
        return self.add(-x, y).norm(dim=dim, keepdim=keepdim)


    def geodesic(self, x, y, t=1.):
        """
            x \oplus (-x \oplus y) \otimes t
        """
        return self.add(x, self.scalar_mul(t, self.add(-x, y)))

    @abstractmethod
    def gyrobarycenter(self, X, weights=None):
        raise NotImplementedError

    def mean(self, X):
        return self.gyrobarycenter(X)[0, :]


class Mobius(HyperbolicDisk):
    """Operations in Poincare space
    """
    def __init__(self, s=1., eps=1e-15):   
        super(Mobius, self).__init__(s=s, eps=eps)
        
    def lambd(self, x, dim=-1, keepdim=True):
        """ Conformal factor at x"""
        norm_x = x.norm(2, dim=dim, keepdim=keepdim).pow(2)
        norm_x.data.clamp_(min=self.min_norm)
        #res = 2. / (1. - self.c * norm_x)
        res = 2. / (1. - norm_x / self.s)
        return res    

    def beta(self, x, dim=-1, keepdim=True):
        """ Conformal factor at x"""
        norm_x = x.norm(2, dim=dim, keepdim=keepdim).pow(2)
        norm_x.data.clamp_(min=self.min_norm)
        #res = 1. / (1. - self.c * norm_x).pow(.5)
        res = 1. / (1. + norm_x / self.s).pow(.5)
        return res            

    def inner(self, x, u, v, dim=-1, keepdim=True):
        """ Inner product in the tanget space at x """
        return self.lambd(x) ** 2 * (u * v).sum(dim=dim, keepdim=keepdim)
    
    def norm(self, x, u, dim=-1, keepdim=True):
        """ Norm in the tanget space at x """
        return self.lambd(x) * u.norm(dim=dim, keepdim=keepdim)

    def add(self, x, y, dim=-1, keepdim=True):
        """
        Definition 3.40 (Mo ̈bius Addition in the Ball). 
            Analytic hyperbolic geometry and Albert Einstein special theory of relativity.

        x + b <= exp_x(P_{0→x}(log_0(b)))
        """
        y = y +  self.eps
        s_sqr = self.s ** 2
        xy = torch.sum(x * y, dim=dim, keepdim=keepdim)
        norm_x = x.norm(2, dim=dim, keepdim=keepdim).pow(2)
        norm_y = y.norm(2, dim=dim, keepdim=keepdim).pow(2)
        num = ((1. + (2. / s_sqr) * xy  + (1. / s_sqr) * norm_y) * x).addcmul(1. - (1. / s_sqr) * norm_x, y)
        den = 1. + (2  / s_sqr) * xy + (1. / s_sqr ** 2) * (norm_x * norm_y) + self.eps
        res = num.div(den)
        #res = torch.where(den == 0, self.eps * torch.ones_like(res), res)
        return self.proj2ball(res)

    def sub(self, x, y, dim=-1, keepdim=True):
        return self.add(-x, y, dim=dim, keepdim=keepdim)

    def coadd(self,x, y, dim=-1, keepdim=True):
        gamma_x_sqr = self.gamma(x, dim=dim, keepdim=keepdim).pow(2)
        gamma_y_sqr = self.gamma(y, dim=dim, keepdim=keepdim).pow(2)
        num = gamma_x_sqr * x + gamma_y_sqr * y
        den = gamma_x_sqr + gamma_y_sqr - 1.
        # avoid division by zero in this way
        return num / den.clamp_min(self.min_norm)

    def cosub(self,x, y, dim=-1, keepdim=True):
        return self.coadd(x, -y, dim=dim, keepdim=keepdim)    
    
    def mat_mul(self, M, x, dim=-1, keepdim=True):
        x = x + self.eps
        #Mx = M.mm(x.t()).t()
        if dim != -1 or M.dim() == 2:
            Mx = torch.tensordot(x, M, dims=([dim], [1]))
        else:
            Mx = torch.matmul(M, x.unsqueeze(-1)).squeeze(-1)

        norm_Mx = Mx.norm(2, dim=dim, keepdim=keepdim) / self.s
        norm_x = x.norm(2, dim=dim, keepdim=keepdim) / self.s
    
        norm_Mx.data.clamp_(min=self.min_norm)
        norm_x.data.clamp_(min=self.min_norm)
        
        result = tanh(norm_Mx.div(norm_x) * atanh(norm_x, self.eps)) * Mx.div(norm_Mx)
        return self.proj2ball(result)

    def distance(self, x, y):
        return 2 * self.s * atanh(self.gyrodistance(x, y) / self.s, self.eps)
    
    def expm(self, x, v, dim=-1, keepdim=True):
        v = v + self.eps
        #s = np.sqrt(self.c)    
        lam = self.lambd(x)
        norm_v_scaled = v.norm(2, dim=dim, keepdim=keepdim) / self.s
        norm_v_scaled.data.clamp_(min=self.min_norm)
        second_term = tanh((0.5 * lam) * norm_v_scaled) * v.div(norm_v_scaled)
        return self.add(x, second_term)

    def logm(self, x, y, dim=-1, keepdim=True):
        x = x + self.eps
        #sqrt_c = np.sqrt(self.c)
        diff = self.add(-x, y)
        norm_diff_scaled = diff.norm(2, dim=dim, keepdim=keepdim) / self.s
        norm_diff_scaled.data.clamp_(min=self.min_norm)
        lam = self.lambd(x)
        return 2. * atanh(norm_diff_scaled, self.eps) * diff.div(norm_diff_scaled * lam)
        
    def expm_zero(self, v, dim=-1, keepdim=True):
        v = v + self.eps
        norm_v_scaled = v.norm(2, dim=dim, keepdim=keepdim) / self.s
        norm_v_scaled.data.clamp_(min=self.min_norm)
        result = tanh(norm_v_scaled) * v.div(norm_v_scaled) 
        return self.proj2ball(result)

    def logm_zero(self, y, dim=-1, keepdim=True):
        y = y + self.eps
        norm_diff_scaled = y.norm(2, dim=dim, keepdim=keepdim) / self.s
        norm_diff_scaled.data.clamp_(min=self.min_norm)
        return atanh(norm_diff_scaled, self.eps) * y.div(norm_diff_scaled)
    
    def rgrad(self, p, d_p, dim=-1, keepdim=True):
        if d_p.is_sparse:
            p_sqnorm = torch.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=dim,
                keepdim=keepdim
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1. - p_sqnorm).pow(2)) / 4.
            n_vals.renorm_(2, 0, 5)
            d_p = torch.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            """ ((1. - p_sqnorm).pow(2) / 4.)
            """
            factor = (1. / self.lambd(p).pow(2)).expand_as(d_p)
            d_p = d_p * factor
        return d_p

    def gyration(self, u, v, w, dim=-1, keepdim=True):
        """ Mobius gyration

        Eq. (3.147) in [1]
        [1] Analytic hyperbolic geometry and Albert Einstein special theory of relativity.

        Parameters
        ----------
        a : tensor
            first point on Poincare ball
        b : tensor
            second point on Poincare ball
        u : tensor
            vector field for operation
        s : float|tensor
            ball negative curvature
        dim : int
            reduction dimension for operations
        Returns
        -------
        tensor
            the result of automorphism
        # non-simplified
        # mupv = -_mobius_add(u, v)
        # vpw = _mobius_add(u, w)
        # upvpw = _mobius_add(u, vpw)
        # return _mobius_add(mupv, upvpw)
        # simplified
        """
        #v = v + eps
        s_2_inv = 1. / self.s ** 2
        s_4_inv = 1. / self.s ** 4

        u2 = u.pow(2).sum(dim=dim, keepdim=keepdim)
        v2 = v.pow(2).sum(dim=dim, keepdim=keepdim)
        uv = (u * v).sum(dim=dim, keepdim=keepdim)
        uw = (u * w).sum(dim=dim, keepdim=keepdim)
        vw = (v * w).sum(dim=dim, keepdim=keepdim)

        A = (-s_4_inv * uw * v2 + s_2_inv) * vw + 2 * s_4_inv * uv * vw
        B = (-s_4_inv * vw * u2) - s_2_inv * uw
        D = 1 + 2 * s_2_inv * uv + s_4_inv * u2 * v2

        return w + 2 * (A * u + B * v) / D.clamp_min(self.min_norm)

    def ptrans(self, x, y, u, dim=-1, keepdim=True):
        lambd_x = self.lambd(x, dim=dim, keepdim=keepdim)
        lambd_y = self.lambd(y, dim=dim, keepdim=keepdim)
        return self.gyration(y, -x, u, dim=dim, keepdim=keepdim) * lambd_x.div(lambd_y)
    
    def ptrans_zero(self, x, y, u, dim=-1, keepdim=True):
        # Parallel transport along the unique geodesic from 0 to x.
        #    logm_x(x ⊕ exp_0(v))
        lambd_x = self.lambd(x, dim=dim, keepdim=keepdim)
        lambd_y = self.lambd(y, dim=dim, keepdim=keepdim)        
        return self.gyration(y, -x, u, dim=dim, keepdim=keepdim) * lambd_x.div(lambd_y)

    def dist2plane(self, x, a, p, keepdim=True, signed=False, dim=-1):
        """
        x: input valuess
        p: bias
        a: l2_normalize(Weights)
        """
        diff = self.add(-p, x)
        diff_norm2 = diff.norm(2, dim=dim, keepdim=keepdim).pow(2)
        sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            sc_diff_a = sc_diff_a.abs()
        a_norm = a.norm(2, dim=dim, keepdim=keepdim)
        num = 2 * self.s * sc_diff_a / self.s
        denom = ((1 - self.s * diff_norm2) * a_norm).clamp_min(self.min_norm)
        return self.s * asinh(num / denom) 

    def geodesic(self, x, y, t):
        """
            x \oplus (-x \oplus y) \otimes t
        """
        return self.add(x, self.scalar_mul(t, self.add(-x, y)))

    def gyrobarycenter(self, X, weights=None, dim=-1, keepdim=True):
        """
            From: Barycenter calculus in Euclidean and Hyperbolic Geometry, a comparative introduyction.
            Eq. (4.16)
        """
        n_samples, n_features = X.shape    
        gamma_X_sqr = self.gamma(X).pow(2)
        weights_ = (np.ones((n_features, n_samples)) / float(n_samples) if (weights is None) else weights)   
        weights = torch.FloatTensor(weights_)
        
        num = weights.mm(gamma_X_sqr * X)
        den = weights.mm(gamma_X_sqr - 0.5)
        return self.scalar_mul(0.5, (num / den))    


    def barycenter_mapping(self, x, coupling, n=None, kind=None): 
        return self.gyrobarycenter(x, coupling)