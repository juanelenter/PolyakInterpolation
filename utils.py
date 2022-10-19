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

'''
class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=self.device, dtype=input.dtype).view(1, -1)
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
'''


class ConjugateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, grad, Omega):
        ctx.save_for_backward(grad)
        return torch.sum(theta * grad, dim=1) - Omega(grad)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output.view(-1, 1), None, None


class FYLoss(torch.nn.Module):

    def __init__(self, weights="average"):
        self.weights = weights
        super(FYLoss, self).__init__()

    def forward(self, theta, y_true):
        self.y_pred = self.predict(theta)
        ret = ConjugateFunction.apply(theta, self.y_pred, self.Omega)

        if len(y_true.shape) == 2:
            # y_true contains label proportions
            ret += self.Omega(y_true)
            ret -= torch.sum(y_true * theta, dim=1)

        elif len(y_true.shape) == 1:
            # y_true contains label integers (0, ..., n_classes-1)

            if y_true.dtype != torch.long:
                raise ValueError("y_true should contains long integers.")

            all_rows = torch.arange(y_true.shape[0])
            ret -= theta[all_rows, y_true]

        else:
            raise ValueError("Invalid shape for y_true.")

        if self.weights == "average":
            return torch.mean(ret)
        else:
            return torch.sum(ret)


def threshold_and_support(z, dim=0):
    """
    z: any dimension
    dim: dimension along which to apply the sparsemax
    """
    sorted_z, _ = torch.sort(z, descending=True, dim=dim)
    z_sum = sorted_z.cumsum(dim) - 1  # sort of a misnomer
    k = torch.arange(1, sorted_z.size(dim) + 1, device=z.device).type(z.dtype).view( torch.Size([-1] + [1] * (z.dim() - 1))).transpose(0, dim)
    support = k * sorted_z > z_sum

    k_z_indices = support.sum(dim=dim).unsqueeze(dim)
    k_z = k_z_indices.type(z.dtype)
    tau_z = z_sum.gather(dim, k_z_indices - 1) / k_z
    return tau_z, k_z


class SparsemaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """
        input (FloatTensor): any shape
        returns (FloatTensor): same shape with sparsemax computed on given dim
        """
        ctx.dim = dim
        tau_z, k_z = threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau_z, min=0)
        ctx.save_for_backward(k_z, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        k_z, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = (grad_input.sum(dim=dim) / k_z.squeeze()).unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(torch.nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class SparsemaxLoss(FYLoss):

    def predict(self, theta):
        return Sparsemax(dim=1)(theta)

    def Omega(self, p):
        return 0.5 * torch.sum((p ** 2), dim=1) - 0.5