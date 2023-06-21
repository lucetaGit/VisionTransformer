import copy

import torch
from torch import nn


def calc_regularization_loss(model):
    loss = 0
    num = 0
    for module in model.modules():
        if hasattr(module, 'regularizer'):
            loss += module.regularizer.calc_loss()
            num += 1
    if num:
        loss /= num

    return loss

def set_regularization(model, args, fn):
    for module in model.modules():
        param = fn(module)
        if not param is None:
            module.regularizer = build_regularization(param, args)

def build_regularization(param, args):
    if args.regularizer == 'kurt':
        regularizer = KurtosisRegularizer(param, args.reg_rate, args.k)
    elif args.regularizer == 'ms':
        regularizer = MaxSquaredRegularizer(param, args.reg_rate)
    else:
        raise Exception(f"Regularizer {args.regularizer} is not found")

    return regularizer


class SetRegulazationFn(object):
    def __init__(self, layer_type: nn.Module, param_type: str):
        self.layer_type = layer_type
        self.param_type = param_type

    def __call__(self, module):
        if isinstance(module, self.layer_type):
            if hasattr(module, self.param_type):
                return getattr(module, self.param_type)
            else:
                raise AttributeError(f"Layer {layer_type} has no attribute {param_type}")
        return None


class Regularizer(object):
    """Regularizer base class."""
    def __init__(self, param:nn.Parameter, rate=1.):
        self.param = param
        self.rate = rate

    """Compute a regularization penalty from an input tensor."""
    def __call__(self, x):
        return 0.0

    def calc_loss(self):
        return self(self.param)


class KurtosisRegularizer(Regularizer):
    def __init__(self, param, rate=1., k=1.8):
        super().__init__(param, rate)
        self.k = k

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        z = (x-mean)/std
        kurt = torch.mean(z**4)
        if kurt < self.k:
            mse = (kurt-self.k)**2
        else:
            mse = 0.
        return self.rate*(mse)

    # def __call__(self, x):
        # mean = torch.mean(x)
        # std = torch.std(x)

        # z = (x-mean)/std
        # kurt = torch.mean(z**4)
        # mse = (kurt-self.k)**2
        # return self.rate*(mse)


class MaxSquaredRegularizer(Regularizer):
    def __init__(self, param, rate=1.):
        super().__init__(param, rate)

    def __call__(self, x):
        return self.rate*(torch.max(x**2))

