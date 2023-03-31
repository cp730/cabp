#############################################
## Artemis                                 ##
#############################################
import pdb

import torch
import torch.nn as nn
import math
from augmenter import NormalAugmenter
from model import BaseModel
from utils import l2norm

from fusion_blocks.CosMo import content_module, style_module, transformers
from fusion_blocks.CLIP4ir import Combiner
from found_fusion_network import *
from artemis_fusion_network import FusionNetwork
from fusion_blocks.VAL import VALAttention
import torch.nn.functional as F


class L2Module(nn.Module):

    def __init__(self):
        super(L2Module, self).__init__()

    def forward(self, x):
        x = l2norm(x)
        return x


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


class ARTEMIS(BaseModel):

    def __init__(self, word2idx, opt):
        super(ARTEMIS, self).__init__(word2idx, opt)
        self.batch_size = opt.batch_size
        # --- modules
        self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())

        self.fusion_net = FusionNetwork(opt, self.Transform_m)
        self.batch_size = opt.batch_size
        self.embed_dim = opt.embed_dim
        self.augmenter = NormalAugmenter(opt)

        self.model_version = opt.model_version
        if self.model_version == "ARTEMIS":
            self.compute_score = self.compute_score_one2all
            self.compute_score_broadcast = self.forwoard

        # --- for heatmap processing
        self.gradcam = opt.gradcam
        self.hold_results = dict()  # holding intermediate results

    def forwoard(self, r, m, t):
        # t_noise = self.augmenter(t)
        A_IS_r, A_IS_t, A_EM_t  = self.fusion_net(r, m, t)

        # noise_score = torch.matmul(reg,t_noise.t())
        Tr_m = self.Transform_m(m)  # 32 *512
        # A_IS_r = self.augmenter(A_IS_r)

        EM_score = (Tr_m.view(self.batch_size, 1, self.embed_dim) * A_EM_t).sum(-1)
        IS_score = (A_IS_r.view(self.batch_size, 1, self.embed_dim) * A_IS_t).sum(-1)
        return EM_score + IS_score

    def compute_score_one2all(self, r, m, t):

        A_IS_r, A_IS_t, A_EM_t = self.fusion_net(r, m, t)  # 32 *512

        Tr_m = self.Transform_m(m)  # 32 *512
        EM_score = (Tr_m.view(self.batch_size, 1, self.embed_dim) * A_EM_t).sum(-1)
        IS_score = (A_IS_r.view(self.batch_size, 1, self.embed_dim) * A_IS_t).sum(-1)


        return EM_score + IS_score


    def arch_parameters(self):
        return self.fusion_net.arch_parameters()

    def clip(self):
        for p in self.arch_parameters():
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

    def genotype(self):
        return self.fusion_net.genotype()

    def apply_attention(self, a, x):
        return l2norm(a * x)


class FoundARTEMIS(BaseModel):

    def __init__(self, word2idx, opt, genotype, criterion):
        super(FoundARTEMIS, self).__init__(word2idx, opt)
        self.batch_size = opt.batch_size
        self.embed_dim = opt.embed_dim
        # --- modules
        self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())

        self.fusion_net = Found_FusionNetwork(args=opt,
                                              criterion=criterion,
                                              genotype=genotype)
        self.model_version = opt.model_version

        # self.txtdecoder = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(self.embed_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.embed_dim, self.embed_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.embed_dim, self.embed_dim)
        # )
        self.instance_norm = nn.InstanceNorm2d(self.embed_dim)
        # self.layer_norm = nn.LayerNorm()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        if self.model_version == "ARTEMIS":
            self.compute_score = self.compute_score
            self.compute_score_broadcast = self.forwoard_search

        self.gradcam = opt.gradcam
        self.hold_results = dict()  #

        # self.gamma = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    def forwoard_search(self, r, m, t):

        Tr_m = self.Transform_m(m)  # 32 *512

        A_IS_r, A_IS_t, A_EM_t = self.fusion_net(r, m, t)  # 32*512, 32*32*512, 32*32*512



        EM_score = (Tr_m.view(self.batch_size, 1, self.embed_dim) * A_EM_t).sum(-1)
        IS_score = (A_IS_r.view(self.batch_size, 1, self.embed_dim) * A_IS_t).sum(-1)
        # positive_aist = torch.zeros(self.batch_size,self.embed_dim)
        # positive_aemt = torch.zeros(self.batch_size, self.embed_dim)
        # for i in range(self.batch_size):
        #     positive_aist[i] = A_IS_t[i][i]
        #     positive_aemt[i] = A_EM_t[i][i]
        # positive_aist = positive_aist.cuda()
        # positive_aemt = positive_aemt.cuda()
        #
        # aisr_scores = torch.matmul(A_IS_r,A_IS_r.t())
        # aist_scores = torch.matmul(A_IS_t, positive_aist.t())
        # Aemt_scores = torch.matmul(positive_aemt, positive_aemt.t())
        # trm_scores = torch.matmul(Tr_m, Tr_m.t())
        # bsc_loss = self.mse(EM_score,IS_score)

        # bsc_loss = (self.mse(aisr_scores, aist_scores) + self.mse(trm_scores,Aemt_scores))/2
        #
        # bsc_loss = (self.mse(aisr_scores, aist_scores) + self.mse(trm_scores, Aemt_scores)) / 2 + self.mse(EM_score,IS_score)

        return EM_score + IS_score


    def compute_score(self, r, m, t):

        A_IS_r, A_IS_t, A_EM_t = self.fusion_net(r, m, t)  # 32 *512

        Tr_m = self.Transform_m(m)  # 32 *512
        EM_score = (Tr_m.view(self.batch_size, 1, self.embed_dim) * A_EM_t).sum(-1)
        IS_score = (A_IS_r.view(self.batch_size, 1, self.embed_dim) * A_IS_t).sum(-1)

        return EM_score + IS_score

    def arch_parameters(self):
        return self.fusion_net.arch_parameters()

    def genotype(self):
        return self.fusion_net.genotype()

    def apply_attention(self, a, x):
        return l2norm(a * x)
