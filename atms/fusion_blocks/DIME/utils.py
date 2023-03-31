import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class Refinement(nn.Module):
    def __init__(self, embed_size):
        super(Refinement, self).__init__()

        self.fc_scale = nn.Linear(embed_size, embed_size)
        self.fc_shift = nn.Linear(embed_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, embed_size)
        self.fc_2 = nn.Linear(embed_size, embed_size)

    def refine(self, query,text_featuers):
        scaling = F.tanh(self.fc_scale(text_featuers)) # 只保留此时text中的重要的内容
        shifting = self.fc_shift(text_featuers)
        modu_res = self.fc_2(F.relu(self.fc_1(query * scaling + shifting)))
        ref_q = modu_res + query

        return ref_q



    def forward(self, image_features, text_features):

        return self.refine(image_features, text_features)







