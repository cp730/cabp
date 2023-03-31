import pdb

import torch
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, args, criterion, optimizer):
        self.opt = args
        self.reg_coefficient = args.reg_coefficient
        self.network_weight_decay = args.weight_decay
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer

    def log_learning_rate(self, logger):
        for param_group in self.optimizer.param_groups:
            logger.info("Architecture Learning Rate: {}".format(param_group['lr']))
            break

    def mlc_loss(self, arch_param):
        c = len(arch_param)
        h,w = arch_param[0].shape
        y_pred_neg = torch.zeros(c,h,w)
        for i in range(c):
            y_pred_neg[i] = arch_param[i]
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        aux_loss = torch.mean(neg_loss)
        return aux_loss


    def step(self, input_valid, txt, target_valid, epoch):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid, txt, epoch)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, txt,  epoch):
        weights = 0 + 50 * epoch / 100
        # ssr_normal = self.mlc_loss(self.model.arch_parameters())
        target_embs = self.model.get_image_embedding(target_valid,epoch)
        scores = self.model(input_valid, txt, target_embs)
        scores = scores.cuda()
        # if epoch > self.opt.warm_up_epochs:
        #     kl_loss = nn.KLDivLoss(reduction="batchmean")
        #     input = F.log_softmax(scores, dim=1)
        #     target = F.softmax(voted_scores, dim=1)
        #     loss_reg = kl_loss(input, target)
        # else:
        #     loss_reg = 0
        # loss = self.criterion(scores) + weights * ssr_normal
        loss = self.criterion(scores)
        # 更新架构参数gamma
        loss.backward()
