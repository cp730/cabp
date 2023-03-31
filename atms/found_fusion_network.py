import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from fusion_types import *

from random import sample
from artemis_fusion_network import AttentionMechanism, FUSION_OPS
from fusion_blocks.DIME.cells import RectifiedIdentityCell
import argparse


# cell and steps in this cell
class Found_NodeCell(nn.Module):
    def __init__(self, node_steps, args, step_genotype):
        super().__init__()
        self.args = args
        self.node_steps = node_steps  # inner steps num

        # self.edge_ops = nn.ModuleList()
        self.node_ops = nn.ModuleList()

        self.num_input_nodes = 2

        # steps in this cell
        inner_steps = step_genotype.inner_steps
        self.compile(inner_steps)

    def compile(self, inner_steps):
        # 找到该cell内部的各个inner_step中的具体的混合操作 Genotype(steps=[StepGenotype(inner_steps=['VAL']), StepGenotype(
        # inner_steps=['CLIP4ir']), StepGenotype(inner_steps=['VAL'])])
        for name in inner_steps:
            node_op = FUSION_OPS[name](self.args)
            self.node_ops.append(node_op)

    def forward(self, x, y, id):
        '''
        如果将来有多个inner steps，这块代码要修改
        '''
        for i in range(self.node_steps):
            out = self.node_ops[i](x, y, id)
        return out


# cell
class Found_FusionNode(nn.Module):
    def __init__(self, node_steps, args, step_genotype):
        '''
        step_genotype:每一个cell内部对应的结构（根据搜索的最优解不同）
        '''
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps  # cell内部的step数

        self.node_cell = Found_NodeCell(node_steps, args, step_genotype)

        self.num_input_nodes = 2
        self.num_keep_edges = 2

    def forward(self, x, y, id):
        out = self.node_cell(x,  y, id)
        return out


# 内部多个cell
class Found_FusionCell(nn.Module):  # 内部有多个cell组成
    def __init__(self, cell_num, args, genotype):
        super().__init__()

        # a list contains multiple cells
        step_nodes = genotype.steps
        self.args = args
        self.bs = args.batch_size
        self.embed_dim = args.embed_dim
        self._compile(step_nodes, args)
        self._steps = cell_num  # cell num
        self.embed_dim = args.embed_dim

    def _compile(self, gene_step_nodes, args):
        self._cell_nodes = nn.ModuleList()
        for gene_step_node in gene_step_nodes:  # for each cell ,node_steps : inner steps num =1
            cell_node = Found_FusionNode(args.node_steps, args, gene_step_node)
            self._cell_nodes.append(cell_node)

    def forward(self, ref_img_features, text_features, tar_img_features):
        out1 = self._cell_nodes[0](ref_img_features,text_features, 0)

        # text_features1 = text_features * A_IS_m
        text_features1 = text_features
        text_features1 = text_features1.view(self.bs, 1, self.embed_dim)
        out2 = self._cell_nodes[1](tar_img_features,  text_features1, 1)

        # text_features2 = text_features * A_EM_m
        text_features2 = text_features
        text_features2 = text_features2.view(self.bs, 1, self.embed_dim)
        out3 = self._cell_nodes[2](tar_img_features, text_features2, 2)

        return out1, out2, out3


class Found_FusionNetwork(nn.Module):

    def __init__(self, args, criterion, genotype):
        super().__init__()

        self._cell_num = args.cell_num  # cell 数
        self.node_steps = args.node_steps  # inner step num
        self._criterion = criterion
        self._genotype = genotype

        # input node number in a cell
        self._num_input_nodes = args.num_input_nodes
        self._num_keep_edges = args.num_keep_edges

        self.cell = Found_FusionCell(self._cell_num, args, self._genotype)

    def forward(self, ref_img_features, text_features,  tar_img_features):
        out1, out2, out3 = self.cell(ref_img_features, text_features, tar_img_features)
        return out1, out2, out3

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

    def genotype(self):
        return self._genotype
