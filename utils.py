import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import resnet

from sps import Sps, SpsL1, SpsL2, ALIG

def map2simplex(mapping = "Softmax"):
    
    assert mapping in ["Softmax", "Taylor", "NormalizedRelu", "TaylorInter", "Sparsemax"], "Mapping not available."
    
    if mapping == "Softmax":
        return nn.LogSoftmax()
    elif mapping == "NormalizedRelu":
        return NormalizedRelu()
    elif mapping == "Taylor":
        return Taylor()
    elif mapping == "TaylorInter":
        return TaylorInter()
    elif mapping == "Sparsemax":
        return Sparsemax()
    
class NormalizedRelu(nn.Module):
    """
    Normalized ReLU map to the simplex: x_i -> log( max{0, x_i / ||x||} )
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        norms = torch.linalg.norm(x, dim = 1)
        out = torch.log( nn.ReLU()( x*(1/norms[:, None]) ) + 1e-12)
        return out
    
class Taylor(nn.Module):
    """
    2nd order Taylor map to the simplex.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        numerator = 1 + x + 0.5*x**2
        denominator = torch.sum(numerator, dim = 1)
        out = torch.log( numerator*(1/denominator[:, None]) + 1e-12 )
        return out

class TaylorInter(nn.Module):
    """
    2nd order Taylor map to the simplex.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        numerator = 0.5 + x + 0.5*x**2
        denominator = torch.sum(numerator, dim = 1)
        out = torch.log( numerator*(1/denominator[:, None]) + 1e-12 )
        return out

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input 
