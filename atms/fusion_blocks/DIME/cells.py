import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from utils import SelfAttention

# from models.Refinement import Refinement
from fusion_blocks.DIME.utils import  Refinement


class RectifiedIdentityCell(nn.Module):
    '''
    保留特征中重要部分的信息
    '''

    def __init__(self, opt):
        super(RectifiedIdentityCell, self).__init__()
        self.keep_mapping = nn.Softmax()

    def forward(self, x):
        emb = self.keep_mapping(x)
        return emb


class IntraModelReasoningCell(nn.Module):
    '''
    对输入的特征（32*512）
    '''

    def __init__(self, opt):
        super(IntraModelReasoningCell, self).__init__()
        self.opt = opt
        self.sa = SelfAttention(opt.embed_size, opt.hid_IMRC, opt.num_head_IMRC)

    def forward(self, inp):
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        sa_emb = self.sa(x)
        if inp.dim() == 4:
            sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
        return sa_emb


class CrossModalRefinementCell(nn.Module):
    def __init__(self, opt):
        super(CrossModalRefinementCell, self).__init__()
        self.refine = Refinement(opt.embed_dim)
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        self.attention = AttentionMechanism(opt)

        self.attention1 = AttentionMechanism(opt)

    def forward(self, image_features, text_features, cell_id):
        attn = self.attention(text_features)
        text_features = text_features * attn
        if cell_id != 0 and cell_id!=3:
            res = torch.zeros(self.bs, self.bs, self.embed_dim)
            for i in range(self.bs):
                text_feats = text_features[i].expand(self.bs,-1)
                # pdb.set_trace()
                res[i] = self.refine(image_features, text_feats)
            res = res.cuda()
            return res
        else:
            rf_pairs_emb = self.refine(image_features, text_features)
            return rf_pairs_emb


class GlobalLocalGuidanceCell(nn.Module):
    def __init__(self, opt):
        super(GlobalLocalGuidanceCell, self).__init__()
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        self.fc_1 = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.fc_2 = nn.Linear(opt.embed_dim, opt.embed_dim)

        self.attention = AttentionMechanism(opt)

    def regulate(self, image_features, text_featuers):
        image_embs_mid = self.fc_1(image_features)
        x = image_embs_mid * text_featuers
        x = F.normalize(x, dim=-2)
        ref_emb = (1 + x) * image_features
        return ref_emb

    def forward(self, img_features, text_features, cell_id):
        attn = self.attention(text_features)
        text_features = text_features * attn
        img_features = img_features * attn
        if cell_id != 0 and cell_id != 3:
            img_rgns = self.regulate(img_features, text_features)
            return img_rgns
        else:
            ref_rgn = self.regulate(img_features, text_features)
            ref_embs = ref_rgn.squeeze(1)
            return ref_embs


class FiLM(nn.Module):

    def __init__(self, opt):
        super(FiLM, self).__init__()

        self.dim = opt.embed_dim
        self.output_dim = 512
        self.bs = opt.batch_size
        self.input_dim = opt.embed_dim
        self.fc = nn.Linear(self.input_dim, 2 * self.dim)
        self.fc_out = nn.Linear(self.dim, self.output_dim)

        self.attention = AttentionMechanism(opt)
    def forward(self, x, y, cell_id):
        y = self.attention(y) * y
        if cell_id != 0 and cell_id != 3:
            outs = torch.zeros(self.bs, self.bs, self.input_dim)
            for i in range(self.bs):
                texts = y[i].expand(32, -1)
                gamma, beta = torch.split(self.fc(texts), self.dim, 1)
                output = gamma * x + beta
                outs[i] = self.fc_out(output)
            outs = outs.cuda()
            return outs
        else:
            gamma, beta = torch.split(self.fc(y), self.dim, 1)
            output = gamma * x + beta
            output = self.fc_out(output)
            # pdb.set_trace()
            return output


class GatedFusion(nn.Module):

    def __init__(self, opt):
        super(GatedFusion, self).__init__()

        self.dim = opt.embed_dim
        self.output_dim = 512
        self.bs = opt.batch_size
        self.input_dim = opt.embed_dim

        self.fc_x = nn.Linear(self.input_dim, self.dim)
        self.fc_y = nn.Linear(self.input_dim, self.dim)
        self.fc_out = nn.Linear(self.dim, self.output_dim)

        self.sigmoid = nn.Sigmoid()

        self.attention = AttentionMechanism(opt)

    def forward(self, image_features, text_features, cell_id):
        text_features = self.attention(text_features) * text_features
        if cell_id != 0 and cell_id != 3:
            outs = torch.zeros(self.bs, self.bs, self.input_dim)
            for i in range(self.bs):
                texts = text_features[i].expand(self.bs,-1)
                out_x = self.fc_x(image_features)
                out_y = self.fc_y(texts)
                gate = self.sigmoid(out_y)
                outs[i] = self.fc_out(torch.mul(gate, out_x))
            outs = outs.cuda()
            return outs
        else:
            out_x = self.fc_x(image_features)
            out_y = self.fc_y(text_features)
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(gate, out_x))

            return output


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
