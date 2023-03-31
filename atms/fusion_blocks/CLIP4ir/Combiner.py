import pdb

import torch
import torch.nn.functional as F
from torch import nn


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, opt):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()

        embed_dim = opt.embed_dim

        self.opt = opt
        self.bs = opt.batch_size
        self.embed_dim = opt.embed_dim
        # self.projection_dim =  embed_dim*2   #  4
        self.hidden_dim = embed_dim  # 8

        # self.text_projection_layer = nn.Linear(embed_dim, self.projection_dim)
        # self.image_projection_layer = nn.Linear(embed_dim, self.projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, embed_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(self.embed_dim * 2, self.hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(self.hidden_dim, 1),
                                            nn.Sigmoid())

        self.attention = self.attention = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Softmax(dim=1)
        )
        self.logit_scale = 100

    def forward(self, image_features,  text_features, cell_id):
        text_features = self.attention(text_features) * text_features
        if cell_id != 0 and cell_id != 3:
            res = torch.zeros(self.bs, self.bs, self.embed_dim)
            for i in range(self.opt.batch_size):
                text_feats = text_features[i].repeat_interleave(32, 0)
                res[i] = self.combiner(image_features, text_feats)
            res = res.cuda()
            return res
        else:
            return self.combiner(image_features, text_features)

    def combiner(self, image_features, text_features):

        text_projected_features = self.dropout1(F.relu(text_features))  # 32 * 512
        image_projected_features = self.dropout2(F.relu(image_features))  # 32 * 512

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)  # 32*1024
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))  # 32*512
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features

        return F.normalize(output)

    def l2norm(self, x):
        """L2-normalize each row of x"""
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        return torch.div(x, norm)


class GeneralizedMeanPooling(nn.Module):
    """
	Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.

	The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

		- At p = infinity, one gets Max Pooling
		- At p = 1, one gets Average Pooling

	The output is of size H x W, for any input size.
	The number of output features is equal to the number of input planes.

	Args:
		output_size: the target output size of the image of the form H x W.
					 Can be a tuple (H, W) or a single H for a square image H x H
					 H and W can be either a ``int``, or ``None`` which means the size will
					 be the same as that of the input.
	"""

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
