"""
A file containing all stochastic activation functions to be used in thesis experimentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

r = random.Random()

class StReLU(nn.Module):
    """
    Stochastic ReLU function that switches between standard ReLU and negative ReLU 
        (x, x < 0 ; 
         0, x >= 0)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, switch_proba = 0.5):
        if r.uniform(0, 1) < switch_proba:
            return F.relu(x)
        else:
            return x * (x < 0)
        
class StRoot(nn.Module):
    """
    Stochastic activation function based on nth root
    """
    def __init__(self, n : int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (n % 1 != 0) or (n < 1): # If passed n is not a positive integer, default to 2 
            n = 2
        self.n = n

    # switch between nth root and nth power
    def forward(self, x, proba=0.5):
        if r.uniform(0, 1) < proba:
            return torch.sign(x) * torch.abs(x) ** (1 / self.n) # allows even roots of negative numbers (mapping to real numbers)
        else:
            return x ** self.n
        
class Stigmoid(nn.Module):
    """
    Stochastic activation function that switches between classic sigmoid and the more computationally efficient hard sigmoid.

    This is inspired by the Stochastic activation function paper from meta fair (https://arxiv.org/pdf/2509.22358) where 
    the authors switch between ReLU and SiLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, proba=0.5):
        if r.uniform(0, 1) < proba:
            return torch.sigmoid(x)
        else:
            return F.hardsigmoid(x)
        
class NMStigmoid(nn.Module):
    """
    Non-monotonic Stochastic Sigmoid activation function that switches between classic sigmoid and a non-monotonic variant to increase expressivity.
    """
    #TODO
    pass

class Bump(nn.Module):
    # TODO
    pass

# TODO

# - run tests with all activation funcs
# - keep track of hyperparameters (learning rate, using batch norm or not, res stream or not, num heads, etc.) and runtime in a table