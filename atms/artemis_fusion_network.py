import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.ImageMath import ops
from torch.autograd import Variable
from utils import l2norm
from fusion_types import FUSION_CHOICES, StepGenotype, Genotype
from fusion_blocks.CosMo.content_module import DisentangledTransformer
from fusion_blocks.CLIP4ir.Combiner import Combiner, GeneralizedMeanPooling
from fusion_blocks.VAL.VALAttention import VAlattention
from fusion_blocks.MAAF.composition_models import Addition, SimpleModelImageOnly, SimpleModelTextOnly, \
    SimpleModelTextOnlyWithAttn
from fusion_blocks.MAAF.fusion import MCB,MLB,Mutan
from fusion_blocks.DIME.cells import GlobalLocalGuidanceCell, CrossModalRefinementCell, FiLM, GatedFusion
from fusion_blocks.RTIC.rtic import RticCompositionModule
from tirg_model import TIRG

FUSION_OPS = {
    'tirg': lambda opt: TIRG(opt),
    # 'MCB':lambda  opt: MCB(opt),
    'MLB':lambda opt:MLB(opt),
    'Mutan':lambda opt:Mutan(opt),
    # 'TIRG2': lambda opt:TIRG2(opt),
    # 'DIFF': lambda opt: FusDiffcg(opt),
    'Combiner': lambda opt: Combiner(opt),
    # 'text_with_attn':lambda opt: SimpleModelTextOnlyWithAttn(opt),
    'add': lambda opt: Addition(opt),
    'artemis_attention': lambda opt: ArtemisAttention(opt),
    'GLG': lambda opt: GlobalLocalGuidanceCell(opt),
    'CMR': lambda opt: CrossModalRefinementCell(opt),
    # 'rtic': lambda opt: RticCompositionModule(opt),
    # 'film': lambda opt: FiLM(opt),
    # 'gf': lambda opt: GatedFusion(opt)
}


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


class ArtemisAttention(nn.Module):

    def __init__(self, opt):
        super(ArtemisAttention, self).__init__()
        self.embed_dim = opt.embed_dim
        self.batch_size = opt.batch_size

        self.Attention_IS = AttentionMechanism(opt)
        self.Attention_EM = AttentionMechanism(opt)
    def forward(self, image_features, text_features, cell_id):
        if cell_id == 2:
            A_EM = self.Attention_EM(text_features)
            return self.apply_attention(A_EM.view(self.batch_size, 1, self.embed_dim), image_features.view(1, self.batch_size, self.embed_dim))
        elif cell_id == 0 or cell_id == 3:
            A_IS = self.Attention_IS(text_features)
            out = self.apply_attention(A_IS, image_features)  # 32 * 512
            return out
        else:
            A_IS = self.Attention_IS(text_features)
            return self.apply_attention(A_IS.view(self.batch_size, 1, self.embed_dim),
                                        image_features.view(1, self.batch_size, self.embed_dim))

    def apply_attention(self, a, x):
        return l2norm(a * x)


class FusionNetwork(nn.Module):
    def __init__(self, args, Tm):
        super(FusionNetwork, self).__init__()

        self._cell_num = args.cell_num  # 3
        self._multiplier = args.multiplier

        self._ops = nn.ModuleList()  # 混合操作
        self.args = args
        self.bs = args.batch_size
        self.embed_dim = args.embed_dim
        self._cell_nodes = nn.ModuleList()  # 一个cell中的step结点列表
        self.num_input_nodes = args.num_input_nodes  # total number of modality features
        self.Transform_m = Tm

        self.ln = nn.LayerNorm(1024)

        self._initialize_cell_nodes(args)

    def _initialize_cell_nodes(self, args):
        for i in range(self._cell_num):  # for each cell
            # num_input = self.num_input_nodes + i
            # step_node = AttentionSumNode(args, num_input)
            cell_node = FusionCell(args.node_steps, args.multiplier, args)
            self._cell_nodes.append(cell_node)

    # 返回cell中每一个step结点的架构参数
    def arch_parameters(self):
        self._arch_parameters = []
        for i in range(self._cell_num):
            self._arch_parameters += self._cell_nodes[i].arch_parameters()
        return self._arch_parameters

    def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]

    def softmax_arch_parameters(self):
        self._save_arch_parameters()
        for p in self._arch_parameters:
            p.data.copy_(F.softmax(p, dim=-1))

    def restore_arch_parameters(self):
        for i, p in enumerate(self._arch_parameters):
            p.data.copy_(self._saved_arch_parameters[i])
        del self._saved_arch_parameters

    def clip(self):
        for p in self.arch_parameters():
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())


    # 将上层的各个特征，按照对应的参数加权和送入到每一个cell中，得到最终的输出
    def forward(self, ref_img_features, text_features,tar_img_features):

        out1 = self._cell_nodes[0](ref_img_features,  text_features, 0)


        text_features1 = text_features
        text_features1 = text_features1.view(self.bs, 1, self.embed_dim)
        out2 = self._cell_nodes[1](tar_img_features, text_features1, 1)

        text_features2 = text_features
        text_features2 = text_features2.view(self.bs, 1, self.embed_dim)
        out3 = self._cell_nodes[2](tar_img_features,  text_features2, 2)



        return out1, out2, out3
    def genotype(self):
        '''
        解析参数
        '''
        gene_steps = []
        for i in range(self._cell_num):
            step_node_genotype = self._cell_nodes[i].node_genotype()
            gene_steps.append(step_node_genotype)

        genotype = Genotype(
            steps=gene_steps
        )
        return genotype


class FusionCell(nn.Module):
    '''
        fusion node  = cell
    '''

    def __init__(self, node_steps, node_multiplier, args):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps  # inner node steps，cell 内部的step结点数
        self.node_multiplier = node_multiplier  # ===混合操作的数量
        self.node_cell = NodeCell(node_steps, node_multiplier, args)  # 初始化这个cell内部的每一个inner step

        self.num_input_nodes = 2  # 每一个cell有两个输入，要被连接到各个step
        self.num_keep_edges = 2  # 每一个step输入有2个

        # self._initialize_betas()
        self._initialize_gammas()

        self._arch_parameters = [self.gammas]

    # def _initialize_betas(self):
    #     k = sum(
    #         1 for i in range(self.node_steps) for n in range(self.num_input_nodes + i))  # 计算出beta有多少个(beta对应的边有多少条)
    #     num_ops = len(STEP_EDGE_PRIMITIVES)
    #     # beta controls node cell arch
    #     self.betas = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)  # 生成的是矩阵k个beta，每个beta有两种选择， 5*2

    # 控制每一个step中的混合操作
    def _initialize_gammas(self):
        k = sum(1 for i in range(self.node_steps))  # 对于cell内的每一个step（我们这里只有一个）
        num_ops = len(FUSION_OPS)
        self.gammas = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)  # 1 * 4

    def forward(self, x, y, id):
        # edge_weights = F.softmax(self.betas, dim=-1)

        node_weights = F.softmax(self.gammas, dim=-1)
        out = self.node_cell(x,  y, node_weights, id)
        return out



    def arch_parameters(self):
        return self._arch_parameters

    def node_genotype(self):

        def _parse(node_weights):
            node_gene = []
            for i in range(self.node_steps):
                W = node_weights[i]
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k

                node_gene.append((FUSION_CHOICES[k_best]))

            return node_gene

        node_weights = F.softmax(self.gammas, dim=-1)
        print("node_weights: ")
        print(node_weights)
        node_gene = _parse(node_weights)

        fusion_gene = StepGenotype(
            inner_steps=node_gene,
        )
        return fusion_gene


# 每个cell 结点内部的step结点
class NodeCell(nn.Module):
    def __init__(self, node_steps, node_multiplier, args):
        super().__init__()

        self.args = args

        self.node_steps = node_steps  # num of step node
        self.node_multiplier = node_multiplier  # == node_steps

        # self.edge_ops = nn.ModuleList()
        self.node_ops = nn.ModuleList()
        self.num_input_nodes = 2
        # self.num_keep_edges = 2

        for i in range(self.node_steps):  # for each step，初始化m

            node_op = StepMixedOp(self.args)
            self.node_ops.append(node_op)  # len = 2 = step数

    def forward(self, x,  y, node_weights, id):

        '''
        cell内每一个step的操作计算过程
        '''
        for i in range(self.node_steps):  # 对于每一个inner step
            out = self.node_ops[i](x,  y, node_weights[i], id)
        return out


class StepMixedOp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._ops = nn.ModuleList()
        # 参数若不统一就不能用这种循环的方式去初始化操作
        for choice in FUSION_CHOICES:
            op = FUSION_OPS[choice](args)
            self._ops.append(op)

    def forward(self, x,  y, weights, id):
        return sum(w * op(x,  y, id) for w, op in zip(weights, self._ops))
