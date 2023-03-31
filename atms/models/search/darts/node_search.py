import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .operations import *
from .node_operations import *
from .genotypes import *
from .model_search import FusionMixedOp

# 每个cell 结点内部的step结点
class NodeCell(nn.Module):
    def __init__(self, node_steps, node_multiplier, args):
        super().__init__()
        
        self.args = args
        
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        
        self.edge_ops = nn.ModuleList()
        self.node_ops = nn.ModuleList()
        
        self.C = args.C
        self.L = args.L
        
        self.num_input_nodes = 2
        # self.num_keep_edges = 2

        for i in range(self.node_steps): # for each inner step
            for j in range(self.num_input_nodes+i):  # 每一个inner step都要和其前面的结点连。low level中，cell内部的每一条边都要初始化为一个mixedOP
                edge_op = FusionMixedOp(self.C, self.L, self.args)
                self.edge_ops.append(edge_op) # len = 5 = 边数
                
        for i in range(self.node_steps):
            node_op = NodeMixedOp(self.C, self.L, self.args)
            self.node_ops.append(node_op) # len = 2 = step数

        if self.node_multiplier != 1:
            self.out_conv = nn.Conv1d(self.C * self.node_multiplier, self.C, 1, 1)
            self.bn = nn.BatchNorm1d(self.C)
            self.out_dropout = nn.Dropout(args.drpt)

        # skip v3 and v4
        self.ln = nn.LayerNorm([self.C, self.L])
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y, edge_weights, node_weights):
        states = [x, y]
        # init_state = self.node_ops[0](x, y, node_weights[0])
        # states.append(init_state)
        offset = 0
        '''
        cell内每一个step的操作计算过程
        '''
        for i in range(self.node_steps): # 对于每一个inner step
            step_input_feature = sum(self.edge_ops[offset+j](h, edge_weights[offset+j]) for j, h in enumerate(states))
            s = self.node_ops[i](step_input_feature, step_input_feature, node_weights[i])
            offset += len(states)
            states.append(s)

        out = torch.cat(states[-self.node_multiplier:], dim=1)
        if self.node_multiplier != 1:
            out = self.out_conv(out)
            out = self.bn(out)
            out = F.relu(out)
            out = self.out_dropout(out)
        
        # skip v4
        out += x
        out = self.ln(out)
        
        return out

# 每一个step结点都是一个fusion node
class FusionNode(nn.Module):
    '''
        fusion node  = cell

    '''
    def __init__(self, node_steps, node_multiplier, args):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps # inner node steps，cell 内部的step结点数
        self.node_multiplier = node_multiplier
        self.node_cell = NodeCell(node_steps, node_multiplier, args) # 初始化这个cell内部的每一个inner step

        self.num_input_nodes = 2
        self.num_keep_edges = 2
        
        self._initialize_betas()
        self._initialize_gammas()

        self._arch_parameters = [self.betas, self.gammas]
        
    def _initialize_betas(self):
        k = sum(1 for i in range(self.node_steps) for n in range(self.num_input_nodes+i)) # 计算出beta有多少个(beta对应的边有多少条)
        num_ops = len(STEP_EDGE_PRIMITIVES)
        # beta controls node cell arch
        self.betas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)# 生成的是矩阵k个beta，每个beta有两种选择， 5*2
    
    def _initialize_gammas(self):
        k = sum(1 for i in range(self.node_steps))
        num_ops = len(STEP_STEP_PRIMITIVES)
        # gamma controls node_step_nodes arch
        self.gammas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True) # 2 * 4
    
    def forward(self, x, y):
        edge_weights = F.softmax(self.betas, dim=-1)
        node_weights = F.softmax(self.gammas, dim=-1)
        out = self.node_cell(x, y, edge_weights, node_weights)        
        return out

    def arch_parameters(self):  
        return self._arch_parameters

    def node_genotype(self):
        def _parse(edge_weights, node_weights):
            edge_gene = []
            node_gene = []

            n = 2
            start = 0
            for i in range(self.node_steps):
                end = start + n
                
                W = edge_weights[start:end]
                edges = sorted(range(i + self.num_input_nodes), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:self.num_keep_edges]
                
                # print("edges:", edges)
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != STEP_EDGE_PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    edge_gene.append((STEP_EDGE_PRIMITIVES[k_best], j))
                    # gene.append((PRIMITIVES[k_second_best], j))

                start = end
                n += 1
                
            for i in range(self.node_steps):
                W = node_weights[i]
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k

                node_gene.append((STEP_STEP_PRIMITIVES[k_best]))

            return edge_gene, node_gene

        concat_gene = range(self.num_input_nodes+self.node_steps-self.node_multiplier, self.node_steps+self.num_input_nodes)
        concat_gene = list(concat_gene)

        edge_weights = F.softmax(self.betas, dim=-1)
        node_weights = F.softmax(self.gammas, dim=-1)
        
        edge_gene, node_gene = _parse(edge_weights, node_weights)

        fusion_gene = StepGenotype(
            inner_edges = edge_gene,
            inner_steps = node_gene,
            inner_concat = concat_gene,
        )
        # print(concat_gene)
        # print(edge_gene)
        # print(node_gene)
        return fusion_gene

if __name__ == '__main__':
    class Args():
        def __init__(self, C, L):
            self.C = C
            self.L = L
            self.drpt = 0.1

    args = Args(16, 8)
    node_cell = NodeCell(2, 1, args)
    fusion_node = FusionNode(2, 1, args)

    a = torch.randn(4, 16, 8)
    b = torch.randn(4, 16, 8)
    # cat_conv_glu = CatConvGlu(16, 8)
    # cat_conv_relu = CatConvRelu(16, 8)

    fusion_node(a, b).shape
    fusion_node.gammas.shape
    fusion_node.node_genotype()







