import pdb

import torch
import torch.nn as nn
from fusion_blocks.CosMo.transformers import AttentionModule
from fusion_blocks.CosMo.style_module import GlobalStyleTransformer2


class DisentangledTransformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_heads = opt.cosmo_num_heads
        if opt.cnn_type == 'resnet50':
            self.pool_dim = feature_size = 2048
        elif opt.cnn_type == 'resnet18':
            self.pool_dim = feature_size = 512
        text_feature_size = opt.embed_dim
        num_heads = opt.cosmo_num_heads

        self.c_per_head = feature_size // num_heads

        assert feature_size == self.n_heads * self.c_per_head

        self.att_module = AttentionModule(feature_size, text_feature_size, num_heads)
        self.att_module2 = AttentionModule(feature_size, text_feature_size, num_heads)
        self.global_styler = GlobalStyleTransformer2(feature_size, text_feature_size)

        self.weights = nn.Parameter(torch.tensor([1., 1.]))
        self.instance_norm = nn.InstanceNorm2d(feature_size)

        self.gemp = GeneralizedMeanPooling(norm=3)
        self.fc = nn.Linear(self.pool_dim, opt.embed_dim)

    def forward(self, x, t, *args, **kwargs):

        normed_x = self.instance_norm(x)  # torch.Size([32, 2048, 7, 7])
        att_out, att_map = self.att_module(normed_x, t,
                                           return_map=True)  # torch.Size([32, 2048, 7, 7]) and torch.Size([32, 8, 49, 49])

        out = normed_x + self.weights[0] * att_out

        att_out2, att_map2 = self.att_module2(out, t, return_map=True)
        out = out + self.weights[1] * att_out2  # torch.Size([32, 2048, 7, 7])

        out = self.global_styler(out, t, x=x)

        out = self.gemp(out).view(-1, self.pool_dim)  # pooling
        out = self.fc(out)
        out = self.l2norm(out)

        return out

    def l2norm(self, x):
        """L2-normalize each row of x"""
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        return torch.div(x, norm)


class GeneralizedMeanPooling(nn.Module):

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
