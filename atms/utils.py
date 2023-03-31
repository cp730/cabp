import pdb

import torch
import torch.nn as nn
import numpy as np
import pickle
import copy
import torch.nn.functional as F
import math


def params_require_grad(module, update):
    for idx, param in enumerate(module.parameters()):
        param.requires_grad = update


def l2norm(x):
    """L2-normalize each row of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)


class SimpleModule(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super(SimpleModule, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_pickle(obj, obj_path):
    obj_file = open(obj_path, "wb")
    pickle.dump(obj, obj_file)
    obj_file.close()


def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AttentionLayer(nn.Module):
    def __init__(self, embed_size, h, is_share=False, drop=0.0):
        super(AttentionLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear]
        else:
            self.linears = clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)

    def forward(self, inp, mask=None):
        # pdb.set_trace()
        inp = inp.cuda()
        nbatches = inp.size(0)
        # pdb.set_trace()
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden, drop=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden)
        self.fc2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class SelfAttentionMap(nn.Module):
    def __init__(self, feature_size, num_heads):
        super().__init__()

        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.W_k = nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=False)
        self.W_q = nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, *args, **kwargs):
        b, c, h, w = x.size()

        keys, queries = self.W_k(x), self.W_q(x)
        keys = keys.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)
        queries = queries.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)

        att_map = torch.bmm(queries.transpose(1, 2), keys) / (self.c_per_head ** 0.5)
        att_map = self.softmax(att_map)  # (b * num_heads, h * w, h * w), torch.sum(att_map[batch_idx][?]) == 1
        att_map = att_map.view(b, self.n_heads, h * w, h * w)

        return att_map

