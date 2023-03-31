import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fusion_types import *

# all node operations has two input and one output, 2C -> C
STEP_STEP_OPS = {
    'Sum': lambda C, L, args: Sum(),
    'ScaleDotAttn': lambda C, L, args: ScaledDotAttn(C, L)
    # 'LinearGLU': lambda C, L, args: LinearGLU(C, args),
    # 'ConcatFC': lambda C, L, args: ConcatFC(C, args)
}

class Sum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return x + y

class LinearGLU(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, 2*C, 1, 1)
        self.bn = nn.BatchNorm1d(2*C)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=1)
        pdb.set_trace()
        out = self.conv(out)
        out = self.bn(out)
        # print(out.shape)
        # apply glu on channel dim
        out = F.glu(out, dim=1)
        out = self.dropout(out)
        pdb.set_trace()
        return out

class ConcatFC(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, C, 1, 1)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x * (torch.tanh(F.softplus(x)))
        return out 

class CatConvMish(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, C, 1, 1)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(args.drpt)
        self.mish = Mish()

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.mish(out)
        out = self.dropout(out)
        return out

class ScaledDotAttn(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, C, L):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(512)

    def forward(self, x, y):
        # trans pose C to last dim

        q = x.t()
        k = y
        v = y.t()
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k) / math.sqrt(d_k)

        attn = F.softmax(scores, dim=-1)
        # pdb.set_trace()
        out = torch.matmul(attn, v) # 512 *32
        # pdb.set_trace()

        out = out.t()
        out = self.dropout(out)
        out = self.ln(out)

        return out

class AttentionMechanism(nn.Module):
	"""
	Module defining the architecture of the attention mechanisms in ARTEMIS.
	"""

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


class NodeMixedOp(nn.Module):
    def __init__(self, C, L, args):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in STEP_STEP_PRIMITIVES:
            op = STEP_STEP_OPS[primitive](C, L, args)
            self._ops.append(op)

    def forward(self, x, y, weights):
        # pdb.set_trace()
        out = sum(w * op(x, y) for w, op in zip(weights, self._ops))
        return out