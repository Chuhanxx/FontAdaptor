import torch 
import random
import time
import torchvision


import torch.nn as nn
import torch.nn.functional as F
from model.transformer import *


from itertools import groupby
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence



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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),)   

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,finalrelu=True,pad=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride,pad=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,pad=pad)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.finalrelu = finalrelu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.finalrelu:
            out = self.relu(out)

        return out


class main_model (nn.Module):
    def __init__(self,block,args):
        self.inplanes = 64
        super(main_model , self).__init__()
        self.args = args
        self.d_model = args.d_model
        self.alphabet = args.alphabet
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,bias=False) #size/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, 2,stride=(2,1))
        self.layer2 = self._make_layer(block, 128, 2,stride=(2,2)) 
        self.decoder = decoder(args)

        self.avgpool = nn.AdaptiveAvgPool2d((4, None))
        self.maxpool = nn.MaxPool2d(2,stride=(2,1))
        self.down = conv1x1(128,64)
        self.linear = nn.Linear(128,28)

        self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down_channel = double_conv(128+64,128)
        self.padding = nn.ZeroPad2d((0,1,0,0))
        self.attn = Attention(self.d_model,self.alphabet)
        self.dropout = torch.nn.Dropout2d(p=0.2, inplace=False)
        if args.TPS:
            self.Transformation = TPS_SpatialTransformerNetwork(
                    F=20, I_size=(args.load_height, args.imgW), I_r_size=(args.load_height, args.imgW), I_channel_num=1)

    def _make_layer(self, block, planes, blocks, stride=1,finalrelu = True):
        # block: name of basic blocks
        # planes: channels of filters
        # blocks: number of ResBlocks, each block has two conv layers 
        # self.inplanes: set to constant number(64), and downsample if input channel != output channel 
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,1),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,1, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(nn.MaxPool2d(stride,stride=stride))
            layers.append(block(self.inplanes, planes,finalrelu = finalrelu))

        return nn.Sequential(*layers)

    def encoder(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1) 
        x_up = self.upsample(x_2) 
        if x_up.shape[-1]!=x_1.shape[-1]:
            x_up= self.padding(x_up)
        x = torch.cat([x_up, x_1], dim=1)
        x = self.down_channel(x) 
        x = self.avgpool(x) 
        x = self.down(x) 
        # making maps
        x = x.view(x.shape[0],-1,x.shape[-1]) 
        x = x/(torch.norm(x,dim=1,keepdim=True) + 1e-6)
        return x


    def forward(self,x,char_seg_labels,img_lengths,char_seg_labels2):
        # encoding
        
        if self.args.TPS:
            x1 = self.normalize(x[1::3,:,:self.d_model])
            x2 = self.normalize(x[0::3,:,:,:self.args.imgW])

            seg = x[2::3,:,:,:self.args.imgW]
            x2,seg = self.Transformation(x2,seg)
            trans_im = x2.clone()
            seg =  F.interpolate(seg,scale_factor =(1,0.5),mode = 'nearest')
            seg_labels = self.relu(self.make_seg_labels(seg.squeeze(1))-1)

        else:
            x = (x/255.0 - 0.5)/0.5
            x1 = x[1::2,:,:]
            x2 = x[0::2,:,:]
        x2 = self.encoder(x2)
        x1 = self.encoder(x1)
        im_len = torch.max(img_lengths[0::2])
        char_lines  = x1[:,:,:self.d_model]
        lines = x2[:,:,:im_len]
        
        sim_map_0 = torch.bmm(lines.transpose(1,2),char_lines).unsqueeze(1)*10 #[N,1,W,H]
        pos_emb_x,pos_emb_y = pos_emb(sim_map_0,img_lengths)
        label_map = torch.stack([char_seg_labels]*sim_map_0.shape[-2],dim=-2)   
        sim_map = torch.cat([sim_map_0*15,label_map.unsqueeze(1),pos_emb_x,pos_emb_y],dim=1) 

        # decoding 
        x,embed = self.decoder(sim_map)
        logits,x_ff = self.attn(x,char_seg_labels2)

        if self.args.TPS:
            return logits, sim_map_0,trans_im,seg_labels
        else: 
            return logits,sim_map_0,x

    def normalize(self,x):
        x = (x/255.0 - 0.5)/0.5
        return x

    def make_seg_labels(self,seg):
        zeros = torch.zeros(seg.shape).cuda()
        noises = torch.rand(seg.shape).cuda()
        per =torch.where(seg==0,noises,zeros)
        seg_labels =torch.mode(per+seg,dim=-2)[0]
        seg_labels[seg_labels<1] = 0
        return seg_labels


    def onehot_char_id(self,char_seg_labels,squeeze =False):
        # input: char_seg_labels [B,H]
        # output: one hot maps correspeonds to character ids [B,C,H]
        num_classes = int(torch.max(char_seg_labels))+1
        onehot = torch.FloatTensor(char_seg_labels.shape[0],num_classes,char_seg_labels.shape[-1]).cuda() #[B,CLASS,H]
        onehot = onehot.fill_(0)+1e-6
        ones = torch.ones(char_seg_labels.unsqueeze(1).shape).cuda() #[B,1,H]
        onehot.scatter_(1, char_seg_labels.unsqueeze(1).long(), ones)

        return onehot 

class decoder (nn.Module):
    def __init__(self,args):
        super(decoder , self).__init__()

        self.heads = 3
        self.d_model = args.d_model

        self.N = 4
        self.inner_d = args.d_model 

        multihead = MultiHeadedAttention(self.heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.inner_d, 0)
        c = copy.deepcopy
        self.embedding = nn.Sequential(nn.Linear(4,16),
            nn.ReLU(inplace=True),
            nn.Linear(16,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,1))
        self.soft_attn_block = Encoder(EncoderLayer(self.d_model, c(multihead), c(ff), 0), self.N)

        for p in self.soft_attn_block.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,sim_map):
        #[B,1,W,H]
        sim_map = torch.transpose(sim_map,1,3)
        embed = self.embedding(sim_map) #[B,H,W,1]
        embed = torch.transpose(embed,1,2) #[B,W,H,1]
        out =  self.soft_attn_block(embed.squeeze(-1)) #[B,W,H]
           
        return out,embed

def generate_labels(labels,width):
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

    return padded_labels.to(torch.float32).unsqueeze(1) #[B,C,H,W]


def pos_emb(x,img_lengths):
    pos_emb_x = torch.ones((x.shape[0],1,x.shape[2])).cuda() # [B,1,W]
    pos_emb_y = torch.ones((x.shape[0],1,x.shape[3])).cuda() 
    for b_idx in range(pos_emb_x.shape[0]):
        im_length_x = int(img_lengths[::2][b_idx])
        im_length_y = int(img_lengths[1::2][b_idx])
        pos_emb_x[b_idx,0,:im_length_x] = torch.linspace(0,1,steps=im_length_x,dtype=torch.float32)
        pos_emb_y[b_idx,0,:im_length_y] = torch.linspace(0,1,steps =im_length_y,dtype=torch.float32)
    pos_emb_x = torch.stack([pos_emb_x]*x.shape[3],dim=-1)
    pos_emb_y = torch.stack([pos_emb_y]*x.shape[2],dim=2)
    return pos_emb_x,pos_emb_y

    
class Attention(nn.Module):
    "attend each time step to glyphs"
    def __init__(self,d_model,alphabet):
        super(Attention, self).__init__()
        self.boundary = nn.Embedding(1,d_model)
        self.bound_m = Variable(torch.rand(1), requires_grad=True)
        self.window_size = 3
        self.conv = nn.Conv2d(1, 2, kernel_size=(1,self.window_size),padding = (0,self.window_size//2))
        self.relu = nn.ReLU(inplace=True)
        self.linear_glu = nn.Linear(d_model,2*d_model)
        self.linear_k = nn.Linear(d_model,d_model)
        self.linear_q =  nn.Linear(d_model,d_model)
        self.num_classes = len(alphabet)


    def forward(self,x,char_seg_labels2):
        choices_0 = self.make_choices(char_seg_labels2)
        x_ff = self.linear_q(x).squeeze(1).transpose(1,2)
        choices_ff = self.linear_k(choices_0)
        choices = choices_ff /torch.norm(choices_ff,dim=-1,keepdim=True)
        x = x_ff / torch.norm(x_ff,dim=-2,keepdim=True)
        out = 15*torch.bmm(choices,x.squeeze(1)).transpose(1,2) # [B,W,num_class]

        return out,x_ff

    def make_choices(self,char_seg_labels,squeeze =False):
        # generate columns correspond to each glyph attended
        # no of classes = blank + 26 + EOS + boundray(learned) == 29
        num_classes = int(torch.max(char_seg_labels))+1
        onehot = torch.FloatTensor(char_seg_labels.shape[0],num_classes,char_seg_labels.shape[-1]).cuda() #[B,CLASS,H]
        onehot = onehot.fill_(0)+1e-6
        ones = torch.ones(char_seg_labels.unsqueeze(1).shape).cuda() #[B,1,H]
        onehot.scatter_(1, char_seg_labels.unsqueeze(1).long(), ones)
        onehot[:,0,:] = self.boundary(torch.LongTensor([0]).cuda())

        return onehot 


class TPS_SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I,seg):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')
        if self.training and torch.sum(seg) !=0:
            seg = F.grid_sample(seg, build_P_prime_reshape,mode = 'nearest', padding_mode='border')
        else: seg = torch.zeros(batch_I_r.shape).cuda()

        return batch_I_r,seg


def GlyphAdaptor(args):
    model = main_model(BasicBlock,args)
    return model
