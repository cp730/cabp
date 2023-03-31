import math
import pdb
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as M
from torch.autograd import Variable

from fusion_blocks.RTIC import act as aa



class RticFusionModule(nn.Module):
    def __init__(self, in_c_img, in_c_text, act_fn):
        super(RticFusionModule, self).__init__()
        assert act_fn in aa.__dict__
        layers = [
            nn.BatchNorm1d(in_c_img + in_c_text),
            nn.__dict__[act_fn](),
            nn.Linear(in_c_img + in_c_text, in_c_img),
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.module(x)


class RticGatingModule(nn.Module):
    def __init__(self, in_c_img, act_fn):
        super(RticGatingModule, self).__init__()
        assert act_fn in aa.__dict__
        layers = [
            nn.Linear(in_c_img, in_c_img),
            nn.BatchNorm1d(in_c_img),
            aa.__dict__[act_fn](),
            nn.Linear(in_c_img, in_c_img),
            nn.Sigmoid(),
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class RticErrorEncodingModule(nn.Module):
    def __init__(self, in_c_img, act_fn):
        super(RticErrorEncodingModule, self).__init__()
        assert act_fn in aa.__dict__
        layers = [
            nn.Linear(in_c_img, in_c_img // 2),
            nn.BatchNorm1d(in_c_img // 2),
            aa.__dict__[act_fn](),
            # nn.Linear(in_c_img // 2, in_c_img // 2),
            # nn.BatchNorm1d(in_c_img // 2),
            # aa.__dict__[act_fn](),
            nn.Linear(in_c_img // 2, in_c_img),
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        """x = fusion_feature (f_f)"""
        return self.module(x) + x


class RticCompositionModule(nn.Module):
    def __init__(self,opt):
        super(RticCompositionModule, self).__init__()
        self.in_c_img = opt.embed_dim
        self.in_c_text = opt.embed_dim
        self.n_blocks = 3
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        # fusion block
        self.fs = RticFusionModule(
            in_c_img=self.in_c_img,
            in_c_text=self.in_c_text,
            act_fn=opt.act_fn,
        )

        # gaiting block
        self.gating = RticGatingModule(
            in_c_img=self.in_c_img,
            act_fn=opt.act_fn,
        )

        # error encoding block
        self.ee = nn.ModuleList()
        for i in range(self.n_blocks):
            ee = RticErrorEncodingModule(
                in_c_img=self.in_c_img,
                act_fn=opt.act_fn,
            )
            self.ee.append(ee)

    def forward(self, image_features,text_features,cell_id):
        if cell_id!=0:
            outs = torch.zeros(self.bs,self.bs,self.embed_dim)
            for i in range(self.bs):
                f = self.fs((image_features, text_features[i].repeat_interleave(32,0)))
                g = self.gating(f)
                for ee in self.ee:
                    f = ee(f)

                out = ((image_features, text_features[i])[0] * g) + (f * (1 - g))
                outs[i]=out
            outs = outs.cuda()
            return outs
        else:

            f = self.fs((image_features, text_features))
            g = self.gating(f)
            for ee in self.ee:
                f = ee(f)
            out = ((image_features, text_features)[0] * g) + (f * (1 - g))
            return out



