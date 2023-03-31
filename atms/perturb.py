import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

import torch
from torch.optim.optimizer import Optimizer, required


class Linf_SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Linf_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Linf_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                #d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball

def Linf_PGD_alpha(model, X, y, epsilon, steps=7, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2 * epsilon / steps)
    with torch.no_grad():
        loss_before = model._loss(X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()

    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss = -model._loss(X, y, updateType='weight')
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()

    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after = model._loss(X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()


def Random_alpha(model,epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.fusion_net.clip()


def Linf_PGD_alpha_RNN(model, X, y, hidden, epsilon, steps=7, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2 * epsilon / steps)
    with torch.no_grad():
        loss_before, _ = model._loss(hidden, X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()

    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss, _ = model._loss(hidden, X, y, updateType='weight')
        loss = -loss
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()

    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after, _ = model._loss(hidden, X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()


def Random_alpha_RNN(model, X, y, hidden, epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.clip()



