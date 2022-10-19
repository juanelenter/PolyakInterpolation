import numpy as np
import torch
import time
import copy


class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.5,
                 gamma=2.0,
                 eta_max=0.2,
                 adapt_flag='smooth_iter',
                 fstar_flag=None,
                 eps=1e-8,
                 centralize_grad_norm=False,
                 centralize_grad=False):
        params = list(params)
        super().__init__(params, {})
        self.eps = eps
        self.params = params
        self.c = c
        self.centralize_grad_norm = centralize_grad_norm
        self.centralize_grad = centralize_grad

        if centralize_grad:
            assert self.centralize_grad_norm is False

        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.state['step'] = 0

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.fstar_flag = fstar_flag

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        # increment step
        self.state['step'] += 1

        # get fstar
        if self.fstar_flag:
            fstar = float(batch['meta']['fstar'].mean())
        else:
            fstar = 0.

        # get loss and compute gradients
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should beNone'

        # save the current parameters:
        grad_current = get_grad_list(self.params, centralize_grad=self.centralize_grad)
        grad_norm = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm)

        if grad_norm < 1e-8:
            step_size = 0.
        else:
            # adapt the step size
            if self.adapt_flag in ['constant']:
                # adjust the step size based on an upper bound and fstar
                step_size = (loss - fstar) / \
                    (self.c * (grad_norm)**2 + self.eps)
                if loss < fstar:
                    step_size = 0.
                else:
                    if self.eta_max is None:
                        step_size = step_size.item()
                    else:
                        step_size = min(self.eta_max, step_size.item())

            elif self.adapt_flag in ['smooth_iter']:
                # smoothly adjust the step size
                step_size = loss / (self.c * (grad_norm)**2 + self.eps)
                coeff = self.gamma**(1./self.n_batches_per_epoch)
                step_size = min(coeff * self.state['step_size'],
                                step_size.item())
            else:
                raise ValueError('adapt_flag: %s not supported' %
                                 self.adapt_flag)

            # update with step size
            sgd_update(self.params, step_size, grad_current)

        # update state with metrics
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = grad_norm.item()

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)


class SpsL1(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=2.,
                 eta_max=None,
                 momentum=0.5,
                 lmbda=0.1,  # eps is 1/lambda
                 centralize_grad_norm=False,
                 centralize_grad=False):
        params = list(params)
        super().__init__(params, {})
        self.lmbda = lmbda
        self.params = params
        self.c = c
        self.centralize_grad_norm = centralize_grad_norm
        self.centralize_grad = centralize_grad

        if centralize_grad:
            assert self.centralize_grad_norm is False

        self.eta_max = eta_max
        self.init_step_size = init_step_size
        self.state['step'] = 0

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['slack'] = 0.
        self.state['velocity'] = [torch.zeros_like(g) for g in params]
        self.momentum = momentum

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        # increment step
        self.state['step'] += 1

        # get loss
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should be None'

        # compute gradients
        grad_current = get_grad_list(self.params, centralize_grad=self.centralize_grad)

        # compute gradient norm
        grad_norm = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm)

        # compute step size
        step_size = max(loss - self.state['slack'] + self.lmbda, 0) / \
            ((grad_norm)**2 + 1)
        step_size2 = loss / (grad_norm)**2

        # update the slack
        self.state['slack'] = max(self.state['slack'] - self.c * self.lmbda + self.c * step_size.item(), 0)
        step_size = self.c * min(step_size2.item(), step_size.item())

        # update velocity
        new_velocity = [self.momentum * v + step_size * g for (v, g) in zip(self.state['velocity'], grad_current)]

        # update params using step size
        sgd_update(self.params, 1., new_velocity)

        # update state with metrics
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm).item()
        self.state['velocity'] = new_velocity

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)


class SpsL2(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=1.,
                 eta_max=None,
                 momentum=0.5,
                 lmbda=0.1,  # eps is 1/lambda
                 centralize_grad_norm=False,
                 centralize_grad=False):
        params = list(params)
        super().__init__(params, {})
        self.lmbda = lmbda
        self.params = params
        self.c = c
        self.centralize_grad_norm = centralize_grad_norm
        self.centralize_grad = centralize_grad

        if centralize_grad:
            assert self.centralize_grad_norm is False

        self.eta_max = eta_max
        self.init_step_size = init_step_size
        self.state['step'] = 0

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['slack'] = 0.
        self.state['velocity'] = [torch.zeros_like(g) for g in params]
        self.momentum = momentum

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        # increment step
        self.state['step'] += 1

        # get loss
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should be None'

        # compute gradients
        grad_current = get_grad_list(self.params, centralize_grad=self.centralize_grad)

        # compute gradient norm
        grad_norm = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm)

        # compute step size
        lambda_hat = 1. / (1 + self.lmbda)
        step_size = max(loss - lambda_hat * self.state['slack'], 0)
        step_size /= (lambda_hat + grad_norm ** 2)

        # update the slack
        self.state['slack'] = self.c*lambda_hat * (self.state['slack'] +
                                            step_size.item()) + 1.0 - self.c

        # now take into account self.c for the w update
        step_size = self.c * step_size.item()

        # update velocity
        new_velocity = [self.momentum * v + step_size * g for (v, g) in zip(self.state['velocity'], grad_current)]

        # update params using step size
        sgd_update(self.params, 1., new_velocity)

        # update state with metrics
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm).item()
        self.state['velocity'] = new_velocity

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)


# utils
# ------------------------------
def compute_grad_norm(grad_list, centralize_grad_norm=False):
    grad_norm = 0.
    for g in grad_list:
        if g is None or (isinstance(g, float) and g == 0.):
            continue

        if g.dim() > 1 and centralize_grad_norm:
            # centralize grads
            g.add_(-g.mean(dim = tuple(range(1,g.dim())), keepdim = True))

        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params, centralize_grad=False):
    grad_list = []
    for p in params:
        g = p.grad
        if g is None:
            g = 0.
        else:
            g = p.grad.data
            if len(list(g.size()))>1 and centralize_grad:
                # centralize grads
                g.add_(-g.mean(dim = tuple(range(1,len(list(g.size())))),
                       keepdim = True))

        grad_list += [g]

    return grad_list


def sgd_update(params, step_size, grad_current):
    for p, g in zip(params, grad_current):
        if isinstance(g, float) and g == 0.:
            continue
        p.data.add_(other=g, alpha=- step_size)



class ALIG(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 eps=1e-5,
                 c=1.0,
                 eta_max=0.1,
                 momentum=0.9,
                 lmbda=1.0,  # eps is 1/lambda
                 centralize_grad_norm=False,
                 centralize_grad=False):
        params = list(params)
        super().__init__(params, {})
        self.lmbda = lmbda
        self.params = params
        self.c = c
        self.eps = eps
        self.centralize_grad_norm = centralize_grad_norm
        self.centralize_grad = centralize_grad

        if centralize_grad:
            assert self.centralize_grad_norm is False

        self.eta_max = eta_max
        self.init_step_size = init_step_size
        self.state['step'] = 0

        self.state['step_size'] = init_step_size
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['velocity'] = [torch.zeros_like(g) for g in params]
        self.momentum = momentum

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        # increment step
        self.state['step'] += 1

        # get loss
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should be None'

        # compute gradients
        grad_current = get_grad_list(self.params, centralize_grad=self.centralize_grad)

        # compute gradient norm
        grad_norm = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm)

        # compute step size
        step_size = min( (loss /  (self.c * (grad_norm)**2 + self.eps)).item(), self.eta_max)

        # update velocity
        new_velocity = [self.momentum * v + step_size * g for (v, g) in zip(self.state['velocity'], grad_current)]

        # update params using step size
        sgd_update(self.params, 1., new_velocity)

        # update state with metrics
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm).item()
        self.state['velocity'] = new_velocity

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)
