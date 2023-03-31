import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from focal_loss_adaptive_gamma import FocalLossAdaptive

class LossModule(nn.Module):

    def __init__(self, opt):
        super(LossModule, self).__init__()
        self.loss_type = opt.loss_type
        self.opt = opt
    def forward(self, scores):

        # build the ground truth label tensor: the diagonal corresponds to
        # correct classification
        GT_labels = torch.arange(scores.shape[0]).long()

        GT_labels = torch.autograd.Variable(GT_labels)
        if torch.cuda.is_available():
            GT_labels = GT_labels.cuda()

        if self.loss_type == 'ce_loss':
            # compute the cross-entropy loss
            loss = F.cross_entropy(scores, GT_labels, reduction='mean')
            # pdb.set_trace()

        elif self.loss_type == 'focal_loss':
            fc_loss = FocalLoss(self.opt)
            loss = fc_loss(scores,GT_labels)
        elif self.loss_type == 'focal_loss_adapt':
            fc_loss = FocalLossAdaptive()
            loss = fc_loss(scores,GT_labels)
        return loss





class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma1 = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**2 * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss



class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)


    def forward(self, an_dist, ap_dist):
        num_samples = an_dist.shape[0]
        y = torch.ones((num_samples, 1)).view(-1)
        pdb.set_trace()
        loss = self.Loss(an_dist - ap_dist, y)

        return loss
