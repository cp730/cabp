import torch
import torch.nn as nn
from model import BaseModel
from utils import l2norm
from fusion_blocks.MAAF.fusion import get_fusion

class ConCatModule(nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x)


class TIRG(nn.Module):

    def __init__(self, opt):
        super(TIRG, self).__init__()
        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        self.fusion = opt.fusion
        if self.fusion == 'base':
            concat_num = 2
        else:
            concat_num = 3
        if opt.fusion== 'hadamard':
            self.fusion = get_fusion(self.fusion)
        elif opt.fusion == 'concat':
            self.fusion = get_fusion(self.fusion, self.embed_dim)
        elif opt.fusion == 'mutan':
            self.fusion = get_fusion(self.fusion,self.embed_dim,self.embed_dim/2,0.1)

        # --- modules
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1, 1]))
        self.gated_feature_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * self.embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * self.embed_dim, self.embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * self.embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * self.embed_dim, 2 * self.embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * self.embed_dim, self.embed_dim))

    def forward(self, image_features, text_features, cell_id):
        if cell_id != 0:
            res = torch.zeros(self.bs, self.bs, self.embed_dim)
            for i in range(self.bs):
                text_feats = text_features[i].repeat_interleave(32, 0)
                fusion = self.fusion(image_features, text_feats)
                x = torch.cat([image_features, text_feats, fusion], dim=1)
                f1 = self.gated_feature_composer(x)
                f2 = self.res_info_composer(x)
                f = torch.sigmoid(f1) * image_features * self.a[0] + f2 * self.a[1]
                res[i] = f
            res = res.cuda()
            return res
        else:
            fusion = self.fusion(image_features, text_features)
            x = torch.cat([image_features, text_features, fusion], dim=1)
            f1 = self.gated_feature_composer(x)
            f2 = self.res_info_composer(x)
            f = torch.sigmoid(f1) * image_features * self.a[0] + f2 * self.a[1]
            return f

