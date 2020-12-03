import torch
from torch import nn
from torch.autograd import Function
from torch.nn.init import xavier_uniform_, uniform_
from manifold import Mobius
from base import asinh


ETA = 1e-9


def euclidean_non_lin(eucl_h, non_lin="id"):
    if non_lin == 'id':
        return eucl_h
    elif non_lin == 'relu':
        return torch.relu(eucl_h)
    elif non_lin == 'tanh':
        return torch.tanh(eucl_h)
    elif non_lin == 'sigmoid':
        return torch.sigmoid(eucl_h)  
    elif non_lin == 'softmax':
        return torch.softmax(eucl_h, dim=1)      
    elif non_lin == 'softplus':
        return nn.functional.softplus(eucl_h) + ETA   
    elif non_lin == 'elu':
        return nn.functional.elu(eucl_h)     
    elif non_lin == 'leaky_relu':
        return nn.functional.leaky_relu(eucl_h)   
    elif non_lin == 'log_softmax':
        return torch.log_softmax(eucl_h, dim=1)         
    return eucl_h   

    
# Applies a non linearity sigma to a hyperbolic h using: 
#   exp_0(sigma(log_0(h)))
def mobius_hyperbolic_non_lin(
    hyp_h, manifold, non_lin="id", hyperbolic_output=False):
    if non_lin == "id":
        return (
            hyp_h if hyperbolic_output 
            else manifold.logm_zero(hyp_h))
    else:
        eucl_h = euclidean_non_lin(
            manifold.logm_zero(hyp_h), non_lin=non_lin)
        return (
            manifold.expm_zero(eucl_h) 
            if hyperbolic_output else eucl_h)   


class MobiusId(nn.Module):
    def __init__(self, manifold=None, hyperbolic_output=True):
        super(MobiusId, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.hyperbolic_output = hyperbolic_output

    def forward(self, x):
        return mobius_hyperbolic_non_lin(
            x, manifold=self.manifold,
            hyperbolic_output=self.hyperbolic_output, 
            non_lin='id')              


class MobiusTanh(nn.Module):
    def __init__(self, manifold=None, hyperbolic_output=True):
        super(MobiusTanh, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.hyperbolic_output = hyperbolic_output

    def forward(self, x):
        return mobius_hyperbolic_non_lin(
            x, manifold=self.manifold,
            hyperbolic_output=self.hyperbolic_output, 
            non_lin='tanh')            


class MobiusSigmoid(nn.Module):
    def __init__(self, manifold=None, hyperbolic_output=True):
        super(MobiusSigmoid, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.hyperbolic_output = hyperbolic_output

    def forward(self, x):
        return mobius_hyperbolic_non_lin(
            x, manifold=self.manifold,
            hyperbolic_output=self.hyperbolic_output, 
            non_lin='sigmoid')          


class MobiusElu(nn.Module):
    def __init__(self, manifold=None, hyperbolic_output=True):
        super(MobiusElu, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.hyperbolic_output = hyperbolic_output

    def forward(self, x):
        return mobius_hyperbolic_non_lin(
            x, manifold=self.manifold,
            hyperbolic_output=self.hyperbolic_output, 
            non_lin='elu')


class MobiusRelu(nn.Module):
    def __init__(self, manifold=None, hyperbolic_output=True):
        super(MobiusRelu, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.hyperbolic_output = hyperbolic_output

    def forward(self, x):
        return mobius_hyperbolic_non_lin(
            x, manifold=self.manifold,
            hyperbolic_output=self.hyperbolic_output, 
            non_lin='relu')


class MobiusLinear(nn.Module):
    def __init__(self, input_features, output_features, manifold=None, 
                 bias=True, left_addition=True):
        super(MobiusLinear, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.input_features = input_features
        self.output_features = output_features
        self.left_addition = left_addition
        # Initiallize 
        self.weight = nn.Parameter(torch.Tensor(output_features, 
                                                input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-.01, .01)        

    def forward(self, x):
        if self.bias is not None:
            if not self.left_addition:
                h = self.manifold.add(
                    self.manifold.mat_mul(self.weight, x),
                    self.manifold.expm_zero(self.bias)
                )
            else:
                h = self.manifold.add(
                    self.manifold.expm_zero(self.bias),
                    self.manifold.mat_mul(self.weight, x),
                )
        else:
            h = self.manifold.mat_mul(self.weight, x)
        return h
          
    def optim_params(self):
        """ To use with GeoOpt
        """
        return [{
            'params': self.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
        }, ]       

    
class MobiusClassificationLayer(nn.Module):
    def __init__(self, input_features, n_classes, manifold=None,
                 bias=True, left_addition=True, signed=True):
        
        super(MobiusClassificationLayer, self).__init__()
        self.manifold = Mobius() if manifold is None else manifold
        self.input_features = input_features
        self.n_classes = n_classes
        self.left_addition = left_addition
        self.signed = signed
        # Initiallize 
        self.weight = nn.Parameter(torch.Tensor(n_classes, input_features))
        self.bias = nn.Parameter(torch.Tensor(n_classes, input_features))

        xavier_uniform_(self.weight)
        self.bias.data.uniform_(-.01, .01)        
        
    def forward(self, x):           
        output = []
        for c in range(self.n_classes):
            a = self.weight[c, :]
            p = self.bias[c, :]
            norm_a_p = self.manifold.norm(p, a) #* self.manifold.s
            decision = self.manifold.dist2plane(
                x=x, p=p, a=a, signed=self.signed) * norm_a_p                  
            output.append(decision)
        return torch.cat(output, dim=1)            
                      
    def optim_params(self):
        """ To use with GeoOpt
        """
        return [{
            'params': self.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
        }, ]         


if __name__ == "__main__":

    mobius = Mobius(eps=1e-15)
    model = nn.Sequential(*[
        MobiusLinear(input_features=2, output_features=2, manifold=mobius),
        MobiusRelu(manifold=mobius),
        MobiusLinear(input_features=2, output_features=2, manifold=mobius),
    ])