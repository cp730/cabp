# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


"""Models for Text and Image Composition."""
import pdb

import torch
import math
import numpy as np
from abc import ABC, abstractmethod
import torch.nn as nn
from .transformer import MultiHeadedAttention, \
    PositionwiseFeedForward, \
    PositionalEncoding, PositionalEncoder, \
    EncoderLayer, Encoder





class SimpleModelImageOnly(torch.nn.Module):
    def __init__(self, opt):
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        torch.nn.Module.__init__(self)
    def forward(self, img_emb,text_emb,cell_id):
        if cell_id!=0 and cell_id!=3:
            res = torch.zeros(self.bs,self.bs,self.embed_dim)
            for i in range(self.bs):
                res[i] = img_emb
            res = res.cuda()
            return res
        else:
            return img_emb


class SimpleModelTextOnly(torch.nn.Module):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
    def forward(self, img_emb, text_emb,cell_id):
        return text_emb

class SimpleModelTextOnlyWithAttn(torch.nn.Module):
    def __init__(self, opt):
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        torch.nn.Module.__init__(self)
        self.attention  = AttentionMechanism(opt)
    def forward(self, img_emb, text_emb,cell_id):
        text_emb = self.attention(text_emb)
        if cell_id != 0:
            res = torch.zeros(self.bs, self.bs, self.embed_dim)
            for i in range(self.bs):
                text = text_emb[i].repeat_interleave(32, 0)
                res[i] = text
            res = res.cuda()
            return res
        else:
            return text_emb




class Addition(torch.nn.Module):
    """Vector addition model."""

    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        self.attention  = AttentionMechanism(opt)

        self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
    def forward(self, img_emb,text_emb,cell_id):
        attn = self.attention(text_emb)
        text_emb = text_emb * attn
        if cell_id!=0 and cell_id!= 3:
            # 将每一个text都要和所有的trg img 相加
            text_emb = text_emb.view(self.bs,1,self.embed_dim)
            img_emb = img_emb.view(1,self.bs,self.embed_dim)
            return self.alpha*img_emb + self.beta*text_emb # 32*32*512
        else:
            return self.alpha*img_emb + self.beta*text_emb


class AttentionMechanism(nn.Module):

    def __init__(self, opt):
        super(AttentionMechanism, self).__init__()

        self.embed_dim = opt.embed_dim
        input_dim = self.embed_dim

        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.attention(x)