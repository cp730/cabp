import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

def activateFunc(x):
    x = torch.tanh(x)
    return F.relu(x)

class Router(nn.Module):
    def __init__(self, opt):
        super(Router, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(opt.embed_dim, opt.embed_dim),
                                    nn.ReLU(True),
                                    nn.Linear(opt.embed_dim, opt.embed_dim))
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):

        x = x.mean(-2)
        pdb.set_trace()
        x = self.mlp(x)
        pdb.set_trace()
        soft_g = activateFunc(x)
        pdb.set_trace()
        return soft_g