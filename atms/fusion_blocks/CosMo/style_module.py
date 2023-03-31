import pdb
from math import sqrt

import torch
from torch import nn




'''
        style module
'''
class GlobalStyleTransformer2(nn.Module):
    def __init__(self, feature_size, text_feature_size, *args, **kwargs):
        super().__init__()
        self.global_transform = EqualLinear(text_feature_size, feature_size * 2)
        self.gate = EqualLinear(text_feature_size, feature_size * 2)
        self.sigmoid = nn.Sigmoid()

        self.init_style_weights(feature_size)

    def forward(self, normed_x, t, *args, **kwargs):
        x_mu, x_std = calculate_mean_std(kwargs['x']) # torch.Size([32, 2048, 1, 1]) and torch.Size([32, 2048, 1, 1])

        gate = self.sigmoid(self.gate(t)).unsqueeze(-1).unsqueeze(-1)
        std_gate, mu_gate = gate.chunk(2, 1)

        global_style = self.global_transform(t).unsqueeze(2).unsqueeze(3)
        gamma, beta = global_style.chunk(2, 1)

        gamma = std_gate * x_std + gamma
        beta = mu_gate * x_mu + beta
        out = gamma * normed_x + beta
        # pdb.set_trace() # torch.Size([32, 2048, 7, 7])
        '''
        torch.Size([32, 2048, 7, 7])这个结果最后要转换32 * 512的维度
        '''
        return out

    def init_style_weights(self, feature_size):
        self.global_transform.linear.bias.data[:feature_size] = 1
        self.global_transform.linear.bias.data[feature_size:] = 0

    @classmethod
    def code(cls) -> str:
        return 'global2'


def reshape_text_features_to_concat(text_features, image_features_shapes):
    return text_features.view((*text_features.size(), 1, 1)).repeat(1, 1, *image_features_shapes[2:])


def calculate_mean_std(x):
    mu = torch.mean(x, dim=(2, 3), keepdim=True).detach()
    std = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False).detach()
    return mu, std


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias=bias)
        linear.weight.data.normal_()
        if bias:
            linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, inputs):
        return self.linear(inputs)


