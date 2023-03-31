#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import pdb
import clip
import torch
import torch.nn as nn
from torch.autograd import Variable

from encoders import EncoderImage, EncoderText,  ClipModel
from utils import params_require_grad, SimpleModule


class BaseModel(nn.Module):
    """
	BaseModel for models to inherit from.
	Simply implement `compute_score` and `compute_score_broadcast`.
	"""

    def __init__(self, word2idx, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.embed_dim = opt.embed_dim
        # use clip model or use normal pretrained backbone
        if opt.use_clip:
            self.clip_model = ClipModel(opt)
            # pdb.set_trace()
            params_require_grad(self.clip_model.clip.transformer, opt.txt_finetune)
            params_require_grad(self.clip_model.clip.visual, opt.img_finetune)
        else:
            self.txt_enc = EncoderText(word2idx, opt)
            params_require_grad(self.txt_enc.embed, opt.txt_finetune)
            if opt.load_image_feature:
                self.img_enc = SimpleModule(opt.load_image_feature, self.embed_dim)
            else:
                self.img_enc = EncoderImage(opt)
                params_require_grad(self.img_enc.cnn, opt.img_finetune)

        # potentially learn the loss temperature/normalization scale at training time
        # (stored here in the code for simplicity)
        self.temperature = nn.Parameter(torch.FloatTensor((opt.temperature,)))

    def get_image_embedding(self, images,epoch):

        if self.opt.use_clip:
            out = self.clip_model.encode_image(images)
        else:
            out = self.img_enc(Variable(images),epoch)
        return out

    def get_txt_embedding(self, sentences, real_sentences, lengths):
        if self.opt.use_clip:
            out = self.clip_model.encode_text(real_sentences)
        else:
            out = self.txt_enc(Variable(sentences), lengths)
        return out

    def compute_score(self, r, m, t):
        raise NotImplementedError

    def compute_score_broadcast(self, r, m, t):
        raise NotImplementedError

    def forward(self, images_src, sentences, images_trg):

        return self.compute_score_broadcast(images_src, sentences, images_trg)

    def get_compatibility_from_embeddings_one_query_multiple_targets(self, r, m, t):

        return self.compute_score(r, m, t)
