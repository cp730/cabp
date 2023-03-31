import pdb

import torch
from torch import nn

class NormalAugmenter(nn.Module):
    '''add Gaussian noisy to feature'''
    def __init__(self,opt):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.instance_norm = nn.InstanceNorm1d(opt.embed_dim)
        self.alpha_scale = opt.alpha_scale
        self.beta_scale = opt.beta_scale

    def forward(self, features):
        res = torch.zeros(32,32,512)
        for i in range(features.shape[0]):
            feat = features[i].squeeze(0) # i th text 和所有trg img

            std, mean = torch.std_mean(feat, dim=1)

            normal_alpha = torch.distributions.Normal(loc=1, scale=std)
            normal_beta = torch.distributions.Normal(loc=mean, scale=std)
            normal_alpha.loc = normal_alpha.loc.cuda()

            alpha = self.alpha_scale * normal_alpha.sample([feat.shape[1]]).transpose(-1, -2)

            beta = self.beta_scale * normal_beta.sample([feat.shape[1]]).transpose(-1, -2)
            feat = feat.unsqueeze(0)
            feat = self.instance_norm(feat).squeeze(0)

            res[i] = alpha * feat + beta
        res = res.cuda()
        return res

    @classmethod
    def code(cls) -> str:
        return 'normal_gaussian'