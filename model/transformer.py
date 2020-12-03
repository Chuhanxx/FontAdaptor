""""
Code edited from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import torch 
import torch.nn as nn
import random
import time
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Variable
from itertools import groupby

import copy


def conv3x3(in_planes, out_planes, stride=1,pad=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=pad, bias=False)

def conv5x5(in_planes, out_planes, stride=1,pad=2):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=pad, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


        
def generate_labels(labels,d_model,max_len):
    # generate a label_map, each pixel with value of the width of glyphs it belongs to
    width = max_len
    width_labels = []
    zeros = torch.zeros
    for i in range(len(labels.tolist())):
        widths = torch.Tensor([sum(1 for i in g) for k,g in groupby(labels[i])]).cuda()
        width_labels.append(torch.repeat_interleave(widths,widths.long()))
    width_labels = torch.stack(width_labels,dim=0) #[B,H]
    width_labels = torch.stack([width_labels]*width,dim=-1) 
    width_labels[width_labels>15] =0
    padded_labels = F.pad(width_labels,(0,0,0,d_model),'constant',0)[:,:d_model,:] 

    return padded_labels.to(torch.float32).transpose(1,2).unsqueeze(-1) #[B,W,H,C]


def onehot_labels(char_seg_labels,max_len):
    # generate onehot labels in shape [B,class,H]
    onehot = torch.FloatTensor(char_seg_labels.shape[0],char_seg_labels.shape[-1],29).cuda() #[B,CLASS,H]
    onehot.zero_()
    ones = torch.ones(char_seg_labels.unsqueeze(1).shape).cuda()
    onehot.scatter_(2, char_seg_labels.unsqueeze(1).long(), ones)
    label_map = torch.stack([onehot]*max_len,dim=1) #[B,class,W,H]
    return label_map



def compute_logits(x,char_seg_labels):
    x = x.unsqueeze(1)  #[B,1,W,H]
    onehot = torch.FloatTensor(x.shape[0],29,x.shape[-1]).cuda() #[B,CLASS,H]
    onehot.zero_()
    ones = torch.ones(char_seg_labels.unsqueeze(1).shape).cuda()
    onehot.scatter_(1, char_seg_labels.unsqueeze(1).long(), ones)
    label_map = torch.stack([onehot]*x.shape[2],dim=2) #[B,class,W,H]
    logits = (x.expand_as(label_map))* label_map
    nonzeros = x.shape[-1] - (logits == 0).sum(dim=-1) 
    ones = torch.ones(nonzeros.shape).cuda().to(torch.int64)
    nonzeros = torch.where(nonzeros <ones,ones,nonzeros)
    logits = logits.sum(-1)/nonzeros.to(torch.float32)

    return logits.transpose(1,2)



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):
        "Implements Figure 2"
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding (nn.Module):
    def __init__(self,dim,max_len=400):
        super(PositionalEncoding,self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len,dim).cuda()
        position = torch.arange(0,max_len).unsqueeze(1).cuda().to(torch.float32)
        div_term = torch.exp((torch.arange(0,dim,2)* -(math.log(10000)/dim)).to(torch.float32) ).cuda()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        pos = Variable(self.pe[:,:x.shape[-1]],
            requires_grad = False).squeeze(0).transpose(0,1)
        pos = torch.stack([pos]*x.shape[0],dim=0)

        return pos

def decoder(args):

    model = main_model(args)

    return model
