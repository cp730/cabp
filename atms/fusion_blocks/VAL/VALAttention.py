import pdb

import torch
from torch import nn
# from fusion_blocks.VAL  import attention
from torch.nn import Conv2d


class SelfAttention(nn.Module):
    def __init__(self, opt):
        super(SelfAttention, self).__init__()
        self.num_head = opt.val_num_heads
        self.image_channel = (512 if (opt.cnn_type == 'resnet18') else 2048)
        hidden_dim = int(self.image_channel / self.num_head)
        self.con2d_11_q = Conv2d(hidden_dim, hidden_dim, 1)
        self.con2d_11_k = Conv2d(hidden_dim, hidden_dim, 1)
        self.con2d_11_v = Conv2d(hidden_dim, hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax()

    def forward(self, features, image_features, num_heads):
        img_shape = image_features.shape  # torch.Size([32, 2048, 7, 7])
        batch_size, h, w, img_channels = img_shape[0], img_shape[2], img_shape[3], img_shape[1]
        location_num = h * w
        hidden_size = img_channels // num_heads

        keys = self.con2d_11_k(features)  # torch.Size([32, 1024, 7, 7])
        values = self.con2d_11_v(features)
        queries = self.con2d_11_q(features)

        keys = keys.view(batch_size, location_num, hidden_size)  # torch.Size([32, 49, 1024])
        values = values.view(batch_size, location_num, hidden_size)
        queries = queries.view(batch_size, location_num, hidden_size)
        bs, dim1, dim2 = values.shape[0], values.shape[1], values.shape[2]
        values = values.view(-1, dim2, dim1)
        att_matrix = torch.bmm(keys, values) / (hidden_size ** 0.5)  # torch.Size([32, 49, 49])

        att_matrix = self.softmax(att_matrix)  # torch.Size([32, 49, 49])

        att_matrix = self.dropout(att_matrix)
        att_out = torch.matmul(att_matrix, queries)
        att_out = att_out.view(batch_size, hidden_size, h, w)  # torch.Size([32, 49, 1024])

        return att_out


class VAlattention(nn.Module):
    def __init__(self, opt):
        super(VAlattention, self).__init__()
        # self.conve2d = Conv2d(1024,512,1)
        self.embed_dim = opt.embed_dim
        self.num_heads = opt.val_num_heads
        self.image_channel = (512 if (opt.cnn_type == 'resnet18') else 2048)
        self.conv2d_cat = Conv2d(self.image_channel + self.embed_dim, self.image_channel, 1)
        self.conv2d_11 = Conv2d(self.image_channel, self.image_channel, 1)
        self.conv2d_77 = Conv2d(1, 1, 7)
        self.self_attention = SelfAttention(opt)
        self.conv2d_11_finl = Conv2d(self.image_channel, self.image_channel, 1)

        self.joint_w = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.self_w = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        if opt.cnn_type == 'resnet50':
            self.pool_dim = 2048
        elif opt.cnn_type == 'resnet18':
            self.pool_dim = 512
        self.gemp = GeneralizedMeanPooling(norm=3)
        self.fc = nn.Linear(self.pool_dim, opt.embed_dim)

    def forward(self, image_features, attn, text_features, id):
        '''
        transform txt feat dim from bs*512 to bs*1*1*512
        image fetaures ：32*size*7*7
      '''
        text_features = text_features * attn
        img_shape = image_features.shape  # torch.Size([32, 2048, 7, 7])
        pdb.set_trace()
        batch_size, h, w, img_channels = img_shape[0], img_shape[2], img_shape[3], img_shape[1]
        text_features = torch.unsqueeze(torch.unsqueeze(text_features, 1), 1)  # 32,1,1,512
        text_features = text_features.view(batch_size, self.embed_dim, 1, 1)  # 32,512,1,1
        text_features = text_features.repeat(1, 1, h, w)  # 32,512,7,7
        vl_features = torch.cat([image_features, text_features], dim=1)  # torch.Size([32, 2560, 7, 7])
        vl_features = self.conv2d_cat(vl_features)  # torch.Size([32, 2048, 7, 7])

        # join attention preservation
        gate_sqz = torch.mean(vl_features, [2, 3], keepdim=True)  # torch.Size([32, 2048, 1, 1])
        att_ch = self.conv2d_11(gate_sqz)  # torch.Size([32, 2048, 1, 1])

        gate_sqz = torch.mean(vl_features, [1], keepdim=True)  # torch.Size([32, 1, 7, 7])
        att_sp = self.conv2d_77(gate_sqz)  # torch.Size([32, 1, 1, 1])

        joint_att = torch.sigmoid(att_ch) * torch.sigmoid(att_sp)  # torch.Size([32, 2048, 1, 1])

        # self-attention transformation
        # 此时，vl_features = 长度为2的tuple，每一个元素为大小torch.Size([32, 1024, 7, 7])的tensor
        vl_features = torch.split(vl_features, int(self.image_channel / self.num_heads), dim=1)

        self_att = []
        # multi-head attention
        for i in range(len(vl_features)):
            self_att.append(self.self_attention(vl_features[i], image_features, self.num_heads))

        self_att = torch.cat(self_att, dim=1)  #

        self_att = self.conv2d_11_finl(self_att)  # torch.Size([32, 2048, 7, 7])

        composite_features = self.joint_w * joint_att * image_features + self.self_w * self_att

        out = self.gemp(composite_features).view(-1, self.pool_dim)  # pooling
        out = self.fc(out)
        out = self.l2norm(out)

        return out  # torch.Size([32, 2048, 7, 7])

    def l2norm(sefl, x):
        """L2-normalize each row of x"""
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        return torch.div(x, norm)


class GeneralizedMeanPooling(nn.Module):


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
