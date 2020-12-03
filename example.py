import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import make_domain_adaptation_toy_data
from hyperbolic_domain_adaptation import (
    mapping_estimation, 
    HyperbolicSinkhornTransport
)

reg_ot = 1e-2
wrapped_function = lambda x: -torch.cosh(x)

n_samples = 300
Xs, Xt, _ = make_domain_adaptation_toy_data(n=n_samples)

###############################################################################
### Check linear mapping estimation
###############################################################################
print("Linear mapping estimation\n")
figsize = [10, 5]
fig, axx = plt.subplots(1, 2, figsize=figsize)

for is_hyperbolic, ax in zip([False, True], axx):
    h_sinkhorn = HyperbolicSinkhornTransport(
        reg_ot=reg_ot, is_hyperbolic=is_hyperbolic,
        wrapped_function=wrapped_function)

    h_sinkhorn.fit(Xs, Xt)
    Xst = h_sinkhorn.transform(Xs)

    ax.scatter(*Xs.data.numpy().T, label='source', marker='X')
    ax.scatter(*Xt.data.numpy().T, label='target', marker='o')
    ax.scatter(*Xst.data.numpy().T, label='transported', marker='X')
    
axx[0].set_title('Euclidean: \nlinear model')
axx[1].set_title('Hyperbolic: \nlinear model')
plt.savefig("linear_estimation.png", bbox_inches='tight')


###############################################################################
### Check different hyperbolic costs
###############################################################################
print("Hyp. mapping estimation with different costs\n")

figsize = [15, 5]
fig, axx = plt.subplots(1, 3, figsize=figsize)

wrapped_functions = {
    '$-cosh \circ d$': lambda x: -torch.cosh(x),
    '$- log \circ (1 + cosh)\circ d$': lambda x: -(torch.cosh(x) + 1).log(),
    '$\pm log \circ cosh \circ d$': lambda x: torch.cosh(x).log(),
    
}

for (w_func_name, w_func), ax in zip(wrapped_functions.items(), axx):
    h_sinkhorn = HyperbolicSinkhornTransport(
        reg_ot=reg_ot, is_hyperbolic=True,
        wrapped_function=w_func)

    h_sinkhorn.fit(Xs, Xt)
    Xst = h_sinkhorn.transform(Xs)

    ax.scatter(*Xs.data.numpy().T, label='source', marker='X')
    ax.scatter(*Xt.data.numpy().T, label='target', marker='o')
    ax.scatter(*Xst.data.numpy().T, label='transported', marker='X')
    
    ax.set_title(w_func_name, fontsize=18)

ax.legend(fontsize=16, markerscale=2)
fig.suptitle('Gyrobarycenter mapping with different costs', y=1.03, fontsize=20)
[plt.savefig(f"gyrobarycenter_map_with_different_costs.{ext}", bbox_inches='tight')
 for ext in ['png', 'pdf']];

###############################################################################
### Check Nearest Neighbors mapping estimation
###############################################################################
print("Nearest Neighbors mapping estimation\n")

figsize = [10, 5]
fig, axx = plt.subplots(1, 2, figsize=figsize)

for is_hyperbolic, ax in zip([False, True], axx):
    h_sinkhorn = HyperbolicSinkhornTransport(
        reg_ot=reg_ot, is_hyperbolic=is_hyperbolic,
        wrapped_function=wrapped_function)

    h_sinkhorn.fit(Xs, Xt)
    Xst = h_sinkhorn.transform(Xs)

    ax.scatter(*Xs.data.numpy().T, label='source', marker='X')
    ax.scatter(*Xt.data.numpy().T, label='target', marker='o')
    ax.scatter(*Xst.data.numpy().T, label='transported', marker='X')
    
axx[0].set_title('Euclidean: \nNearest neighbors model')
axx[1].set_title('Hyperbolic: \nNearest neighbors model')
[plt.savefig(f"nearest_neighbors_estimation.{ext}", bbox_inches='tight') 
 for ext in ['pdf', 'png']];


###############################################################################
### Check Hyperbolic Neural Network mapping estimation
###############################################################################
from mobius_neural_network import MobiusLinear, MobiusRelu

print("Hyp. Neural Network mapping estimation\n")


model = torch.nn.Sequential(*[
    MobiusLinear(2, 100),
    MobiusRelu(),
    MobiusLinear(100, 100),
    MobiusRelu(),
    MobiusLinear(100, 100),
    MobiusRelu(),
    MobiusLinear(100, 2),    
])
lr = 1e-3

plt.figure()
model, coupling = mapping_estimation(Xs, Xt, lr=lr, 
                                     wrapped_function=wrapped_function,
                                     transport_model=model, verbose=True)

plt.scatter(*Xs.data.numpy().T, label='source', marker='X')
plt.scatter(*Xt.data.numpy().T, label='target', marker='o')
plt.scatter(*model(Xs).data.numpy().T, label='transported', marker='X') 

[plt.savefig(f"HNN_mapping_estimation.{ext}", bbox_inches='tight') 
  for ext in ['pdf', 'png']];