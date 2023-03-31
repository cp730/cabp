#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pdb
import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torchvision
from Router import Router
from torch.nn.utils.rnn import pack_padded_sequence

from utils import l2norm,SelfAttentionMap
from config import TORCH_HOME, GLOVE_DIR


def get_cnn(arch):
    return torchvision.models.__dict__[arch](pretrained=True)


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


class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()

        # general parameters
        embed_dim = opt.embed_dim
        self.gradcam = opt.gradcam
        self.opt = opt
        self.bs = opt.batch_size
        self.cnn = get_cnn(opt.cnn_type)
        self.cnn_dim = self.cnn.fc.in_features
        self.pool_dim = self.cnn_dim  # for avgpool resnet, cnn output and pooling output have the same dim
        # replace the avgpool and the last fc layer of the CNN with identity
        # then stack new pooling and fc layers at the end
        self.cnn.avgpool = nn.Sequential()
        self.cnn.fc = nn.Sequential()
        self.gemp = GeneralizedMeanPooling(norm=3)
        self.gemp1 = GeneralizedMeanPooling(norm=3, output_size=7)

        self.fc = nn.Linear(3840, embed_dim)
        self.fc1 = nn.Linear(2048,embed_dim)

        self.dp = nn.Dropout(p=0.25)



    @property
    def dtype(self):
        return self.cnn.conv1.weight.dtype


    def forward(self,images,epoch):
        x = images

        for index, layer in enumerate(self.cnn.children()):
            x = layer(x)
            if index == 4:
                out1 = x
            if index == 5:
                out2 = x

            if index == 6:
                out3 = x

            if index == 7:
                out_7x7 = x
                # out_g = self.attentional_pooling(out_7x7).view(-1, self.pool_dim)
                out_g = self.gemp(out_7x7).view(-1, self.pool_dim)
                out_g = self.fc1(out_g)
                out_g = l2norm(out_g)

        out1 = self.gemp1(out1)
        out2 = self.gemp1(out2)
        out3 = self.gemp1(out3)
        out = torch.cat([out1, out2, out3, out_7x7], dim=1)
        # attn =self.sa(out).squeeze(1)
        # V = out.view(n,3840,49).transpose(1,2)
        # out = torch.matmul(attn,V).view(n,3840,7,7) +self.beta*out
        # pdb.set_trace()
        out = self.gemp(out).view(-1, 3840)
        out = self.fc(out)
        out = l2norm(out)
        out = out + (epoch / 80) * out_g

        return out




    def activations_hook(self, grad):
        """ hook for the gradients of the activations """
        self.gradients = grad

    def register_activations(self, activations):
        self.activations = activations

    def get_gradient(self):
        """ gradient extraction """
        return self.gradients

    def get_activation(self):
        """ activation extraction """
        return self.activations

    def _upsample_add(self, x, y):

        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

class ClipModel(nn.Module):
    def __init__(self,opt):
        super(ClipModel, self).__init__()
        self.opt = opt
        self.cnn_type = opt.clip_img_encoder_type
        self.clip, self.transforms = clip.load(self.cnn_type)
        self.cnn_dim = self.clip.visual.attnpool.c_proj.out_features  # clip 模型最后一层的输出维度
        self.pool_dim = self.cnn_dim
        self.fc1 = nn.Linear(self.pool_dim, opt.embed_dim)
        self.fc2 = nn.Linear(self.pool_dim, opt.embed_dim)
        self.dtype = torch.float32
        # self.modify_pamams_type()

    def encode_image(self,images):
        images = torch.tensor(images)
        out = self.clip.encode_image(images).type(self.dtype)

        out = self.fc1(out)
        out = l2norm(out)

        return out

    def encode_text(self,real_text):

        text_inputs = clip.tokenize(real_text, context_length=77, truncate=True).cuda()

        with torch.no_grad():
            out = self.clip.encode_text(text_inputs).type(self.dtype)
        out = self.fc2(out)
        out = l2norm(out)

        return out


    def image_model_parameters(self, include_scratch=True):
        if not include_scratch:
            return self.clip.visual.parameters()
        try:
            return self.clip.visual.parameters()
        except AttributeError:
            return []

    def image_model_fc_parameters(self):
        try:
            return self.clip.visual.fc.parameters()
        except AttributeError:
            return []

    def text_model_parameters(self, include_scratch=True):
        if not include_scratch:
            return self.clip.transformer.pretrained_parameters()
        try:
            return self.clip.transformer.parameters()
        except AttributeError:
            return []
    def get_params(self):
        # create optimizer
        param_dicts = []
        gathered_params = set()
        # apply learning rate adjustments for model components
        image_fc = [p for p in self.fc1.parameters()]
        text_fc = [p for p in self.fc2.parameters()]
        gathered_params.update(image_fc)
        param_dicts.append({
            'params': image_fc,
            'lr': self.opt.lr
        })
        param_dicts.append({
            'params': text_fc,
            'lr': self.opt.lr
        })
        image_params = self.image_model_parameters()
        # other_img = [p for p in image_params if p not in gathered_params]
        # gathered_params.update(other_img)
        param_dicts.append({
            'params': image_params,
            'lr': self.opt.PRETRAINED_WEIGHT_LR_FACTOR_IMAGE * self.opt.LEARNING_RATE
        })

        text_params = self.text_model_parameters()
        # text_params = [p for p in text_params]
        # gathered_params.update(text_params)
        param_dicts.append({
            'params': text_params,
            'lr': self.opt.PRETRAINED_WEIGHT_LR_FACTOR_TEXT * self.opt.LEARNING_RATE
        })

        return param_dicts








class EncoderText(nn.Module):

    def __init__(self, word2idx, opt):
        super(EncoderText, self).__init__()

        self.opt = opt
        # self.clip_model = clip_model
        wemb_type, word_dim, embed_dim = \
            opt.wemb_type, opt.word_dim, opt.embed_dim

        self.txt_enc_type = opt.txt_enc_type
        self.embed_dim = embed_dim
        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        # Sentence embedding
        if self.txt_enc_type == "bigru":
            self.sent_enc = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)
            self.forward = self.forward_bigru
        elif self.txt_enc_type == "lstm":
            self.lstm_hidden_dim = opt.lstm_hidden_dim
            self.sent_enc = nn.Sequential(
                nn.LSTM(word_dim, self.lstm_hidden_dim),
                nn.Dropout(p=0.1),
                nn.Linear(self.lstm_hidden_dim, embed_dim),
            )
            self.forward = self.forward_lstm

        self.init_weights(wemb_type, word2idx, word_dim)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type is None:
            print("Word embeddings randomly initialized with xavier")
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=GLOVE_DIR)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # Get word embeddings + keep track of missing words
            missing_words = []
            for word, idx in word2idx.items():
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    @property
    def dtype(self):
        return self.embed.weight.data.dtype

    @property
    def device(self):
        return self.embed.weight.data.device



    def forward_bigru(self, x,  lengths):

        # embed word ids to vectors
        wemb_out = self.embed(x)

        # for pytorch >= 1.7, length.device == 'cpu' (but it worked as a gpu variable in 1.2)
        lengths = lengths.cpu()

        # forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.sent_enc.flatten_parameters()

        _, rnn_out = self.sent_enc(packed)
        # reshape output to (batch_size, hidden_size)
        rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1, self.embed_dim)

        out = l2norm(rnn_out)
        # pdb.set_trace()
        return out

    def forward_lstm(self, x,  lengths):

        # embed word ids to vectors
        wemb_out = self.embed(x)  # size (batch, max_length, word_dim)
        wemb_out = wemb_out.permute(1, 0, 2)  # size (max_length, batch, word_dim)

        # lstm
        batch_size = wemb_out.size(1)
        first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                        torch.zeros(1, batch_size, self.lstm_hidden_dim))
        if torch.cuda.is_available():
            first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        lstm_output, last_hidden = self.sent_enc[0](wemb_out, first_hidden)

        # extract features
        text_features = []
        for i in range(batch_size):
            text_features.append(lstm_output[:, i, :].max(0)[0])
        text_features = torch.stack(text_features)

        # output
        out = self.sent_enc[1:](text_features)
        out = l2norm(out)
        return out




