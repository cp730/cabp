import pdb

import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux

import models.search.train_searchable.fashionIQ as tr
import encoders

import numpy as np

from FusionNetWork import FusionNetwork
# from .darts.model import Found_FusionNetwork
from models.search.plot_genotype import Plotter
from architect import Architect
from utils import params_require_grad

from torch.autograd import Variable
def train_darts_model( dataloaders, args, device, logger,word2idx,vocab):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batch_size
    criterion = torch.nn.BCEWithLogitsLoss()
    # model to train
    model = Searchable_Image_Text_Net(args, criterion,word2idx)
    params = model.central_params()  # 返回网络（reshape layer，fusion net和classfier）的参数

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=args.weight_decay)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    # arc_opt,更新的是架构参数。在验证集上更新
    arch_optimizer = op.Adam(model.arch_parameters(),
                             lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model)
    # pdb.set_trace()
    model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer)

    plotter = Plotter(args)
    # search
    best_f1, best_genotype = tr.train_fashionIQ_track_f1(model, architect,
                                                      criterion, optimizer, scheduler, dataloaders,

                                                      device=device,
                                                      num_epochs=args.num_epochs,
                                                      parallel=args.parallel,
                                                      logger=logger,
                                                      plotter=plotter,
                                                      args=args,
                                                      vocab=vocab,
                                                      )

    return best_f1, best_genotype


# create the whole net (two backbones for extract features,reshape layers ,fusion net,and classifier )needed to be searched
class Searchable_Image_Text_Net(nn.Module):
    def __init__(self, args, criterion,word2idx):
        super().__init__()

        self.args = args
        self.criterion = criterion
        # 两个网络分别用于image和text
        self.imagenet = encoders.EncoderImage(args)
        params_require_grad(self.imagenet, args.img_finetune) # set requireg

        self.textnet = encoders.EncoderText(word2idx,args)
        params_require_grad(self.textnet, args.txt_finetune)
        # self.imagenet = mmimdb.GP_VGG(args)
        # self.textnet = mmimdb.MaxOut_MLP(args)

        self.reshape_layers = self.create_reshape_layers(args)  # a list of net ：reshape_layers把输入维度reshape为相同
        # pdb.set_trace()
        self.multiplier = args.multiplier  # cell output conca， default=2 ？
        self.steps = args.steps  # 'cell steps', default=2，有几个cell
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes  # total number of modality features，default = 6
        self.num_keep_edges = args.num_keep_edges  # cells and steps will have 2 input edges ，deefault = 2

        self._criterion = criterion

        self.fusion_net = FusionNetwork(steps=self.steps, multiplier=self.multiplier,
                                        num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                        args=self.args,
                                        criterion=self.criterion)

        self.central_classifier = nn.Linear(512,
                                            args.num_outputs)

    '''
        把各层image features和text features统一维度，使得image 和 text features的维度能够满足每一个cell的输入的维度
        每一个cell的输入的维度，也就是每一个step的输入的维度，也就是每一个fusion模块的输入的维度
        
        对于每一个要reshape的特征，都有一个对应的ReshapeInputLayer_fashionIQ对象
    '''
    def create_reshape_layers(self, args):
        C_ins = [512, 512, 512, 512, 64, 128]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)): # for each features
            reshape_layers.append(aux.ReshapeInputLayer_fashionIQ(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    # def get_image_embedding(self, images):
    #     return self.img_enc(Variable(images))
    #
    # def get_txt_embedding(self, sentences, lengths):
    #     return self.txt_enc(Variable(sentences), lengths)


    def forward(self, tensor_tuple,length):
        text, image = tensor_tuple

        # 从backbone中提取的四个阶段的特征
        # apply net on input image
        image_features = self.imagenet(image)
        image_features = image_features[-1] # imagenet返回的是四个中间层的特征和一个最后层的特征，这里把最后层的特征去了

        # apply net on input skeleton
        text_features = self.textnet(text,length) # 32* 512
        text_features = text_features[0]
        # image_features:torch.Size([32, 256, 56, 56])
        # text_features: torch.Size([32, 512, 28, 28])
        input_features = [image_features] + [text_features]
        # input_features = list(image_features) + list(text_features)

        # pdb.set_trace()
        # input_features = self.reshape_input_features(input_features)

        # pdb.set_trace()
        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self.fusion_net.genotype()

    def central_params(self):
        central_parameters = [
            {'params': self.imagenet.parameters()},
            {'params': self.textnet.parameters()},
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

    def arch_parameters(self):
        return self.fusion_net.arch_parameters()


class Found_Image_Text_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.imagenet = mmimdb.GP_VGG(args)
        self.textnet = mmimdb.MaxOut_MLP(args)
        self._genotype = genotype

        self.reshape_layers = self.create_reshape_layers(args)

        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges

        self.criterion = criterion

        self.fusion_net = Found_FusionNetwork(steps=self.steps, multiplier=self.multiplier,
                                              num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                              args=self.args,
                                              criterion=self.criterion,
                                              genotype=self._genotype)

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 512, 512, 512, 64, 128]
        reshape_layers = nn.ModuleList()

        input_nodes = []
        for edge in self._genotype.edges:
            input_nodes.append(edge[1])
        input_nodes = list(set(input_nodes))

        for i in range(len(C_ins)):
            if i in input_nodes:
                reshape_layers.append(aux.ReshapeInputLayer_MMIMDB(C_ins[i], args.C, args.L, args))
            else:
                # here the reshape layers is not used, so we set it to ReLU to make it have no parameters
                reshape_layers.append(nn.ReLU())

        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):
        text, image = tensor_tuple

        # apply net on input image
        image_features = self.imagenet(image)
        image_features = image_features[0:-1]

        # apply net on input skeleton
        text_features = self.textnet(text)
        text_features = text_features[0:-1]

        input_features = list(image_features) + list(text_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)
