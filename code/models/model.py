import torch
import torch.nn as nn
import torch.nn.functional as F
from fontTools.misc.classifyTools import Classifier
from torchvision import models
from sklearn.decomposition import IncrementalPCA
from .attn import FullAttention, ProbAttention,AttentionLayer,WSASAttention
from .embed import DataEmbedding
from .encoder import Encoder, EncoderLayer, ConvLayer, MultiScaleConvLayer

class ASAFormer(nn.Module):
    def __init__(self, num_classes, feature_dim, len_window, d_model, n_head, e_layers,
                 factor=5,  d_ff=512,
                 dropout=0.0, attn='wsasa', embed='fixed', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(ASAFormer, self).__init__()

        self.attn = attn
        self.output_attention = output_attention

        self.dropout = nn.Dropout(dropout)

        # Attention
        Attn = WSASAttention

        # Encoding
        self.enc_embedding = DataEmbedding(feature_dim, d_model, embed, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_head, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                MultiScaleConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection1 = nn.Linear(int(len_window * d_model / pow(2,e_layers-1)), d_model, bias=True)
        self.projection2 = nn.Linear(d_model, num_classes, bias=True)
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax()

    def forward(self, x_enc,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # bs*len_window/4*d_model
        enc_out = enc_out.reshape(enc_out.size(0), -1)
        out = self.projection1(enc_out)
        out = self.act1(out)
        out = self.projection2(out)
        # out = self.act2(out)

        if self.output_attention == True:
            return out, attns

        elif self.output_attention == False:
            return out