import torch
import torch.nn as nn

from .base import BaseQuantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        assert bit_type.signed
        super(LsqQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None
        self.s_grad_scale = None

    def update_quantization_params(self, *args, **kwargs):
        scale, _ = self.observer.get_quantization_params(*args, **kwargs)
        self.scale = nn.Parameter(scale, requires_grad=True)

    def quant(self, inputs, scale=None, zero_point=None):
        assert zero_point is None
        
        lower = self.bit_type.lower_bound
        upper = self.bit_type.upper_bound

        if scale is None:
            scale = self.scale
        if self.s_grad_scale is None:
            self.s_grad_scale = 1.0/((upper*inputs.numel())**0.5)
        scale = grad_scale(scale, self.s_grad_scale)
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)

        outputs = inputs / scale
        outputs = outputs.clamp(lower, upper)
        outputs = round_pass(outputs)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        assert zero_point is None
        assert not self.scale is None

        lower = self.bit_type.lower_bound
        upper = self.bit_type.upper_bound

        if scale is None:
            scale = self.scale
        if self.s_grad_scale is None:
            self.s_grad_scale = 1.0/((upper*inputs.numel())**0.5)
        scale = grad_scale(scale, self.s_grad_scale)
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)

        outputs = inputs * scale
        return outputs

    def forward(self, inputs):
        lower = self.bit_type.lower_bound
        upper = self.bit_type.upper_bound

        scale = self.scale
        if self.s_grad_scale is None:
            self.s_grad_scale = 1.0/((upper*inputs.numel())**0.5)
        scale = grad_scale(scale, self.s_grad_scale)
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)

        outputs = inputs / scale
        outputs = outputs.clamp(lower, upper)
        outputs = round_pass(outputs)
        outputs = outputs * scale
        
        return outputs

