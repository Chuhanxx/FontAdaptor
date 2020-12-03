import os
import sys
import glob
import torch

import collections
import editdistance
import torchvision

import numpy as np 
import _pickle as cp
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional

from PIL import Image,ImageOps
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageFile, ImageEnhance,ImageFilter,ImageFile
from mpl_toolkits.axes_grid1 import make_axes_locatable,axes_size



ImageFile.LOAD_TRUNCATED_IMAGES = True

class strLabelConverter(object):
	"""Convert between str and label.
	Args:
		alphabet (str): set of the possible characters.
	"""

	def __init__(self, alphabet,upper=False):
		self.upper = upper
		if not self.upper:
			alphabet = alphabet.lower()
		self.alphabet = alphabet

		self.dict = {}
		for i, char in enumerate(alphabet):
			self.dict[char] = i 

	def encode(self, text):
		"""Support batch or single str.
		Args:
			text (str or list of str): texts to convert.
		Returns:
			list of ind
		"""
		if isinstance(text, str):

			string = [char for char in text if char in self.alphabet]
			string = ''.join(string)
			string = string.replace('  ',' ')
			text = [self.dict[char] for char in string if char in self.alphabet]

			length = [len(text)]
		elif isinstance(text, collections.Iterable):
			length = [len(s) for s in text]
			text = ''.join(text)
			text, _ = self.encode(text)

		return text, length

	def decode(self, t, length,type='ce', raw=False):

		"""Decode encoded texts back into strs.
		Args:
			torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
			torch.IntTensor [n]: length of each text.
		Raises:
			AssertionError: when the texts and its length does not match.
		Returns:
			text (str or list of str)
		"""

		# if len(self.alphabet) == 29 and type == 'ce':
		#     t = t +1

		stop_token = len(self.alphabet)-1
		if length.numel() == 1:
			length = length[0]
			assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
			if raw:
				return ''.join([self.alphabet[i] for i in t])
			else:
				char_list = []
				for i in range(length):
					if t[i] != stop_token :
						char_list.append(self.alphabet[t[i]])
					else:
						break
				return ''.join(char_list)
		else:
			# batch mode
			assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
			texts = []
			index = 0
			for i in range(length.numel()):
				l = length[i]
				texts.append(
					self.decode(
						t[index:index + l], torch.IntTensor([l]),type=type, raw=False))
				index += l
			return texts


	def decode_CTC (self,preds,type):
		# convert output probs (with break token) to list of strings
		# outputs [batch,seq_len,classes], after softmax
		dec_preds = []
		if type == 'pred':
			if len(preds.shape) ==1: 
				for t in range(len(preds)):
					if p[t] == len(self.alphabet)-1 :
						break
					if p[t] != 0 and ((t == 0) or (t != 0 and p[t] != p[t - 1])):
						pred.append(self.alphabet[int(p[t])])
				dec_preds.append(''.join(pred))
			for i, p in enumerate(preds):
				pred = ['']
				for t in range(len(p)):
					if p[t] == len(self.alphabet)-1 :
						continue
					if p[t] != 0 and ((t == 0) or (t != 0 and p[t] != p[t - 1] )):
						pred.append(self.alphabet[int(p[t])])
				dec_preds.append(''.join(pred))

		if type == 'label':
			if len(preds.shape) ==1: 
				for t in range(len(preds)):
					if p[t] == len(self.alphabet)-1 :
						break
					if p[t] != 0 and ((t == 0) or (t != 0 )):
						pred.append(self.alphabet[int(p[t])])
				dec_preds.append(''.join(pred))
			for i, p in enumerate(preds):
				pred = ['']
				for t in range(len(p)):
					if p[t] == len(self.alphabet)-1 :
						break
					if p[t] != 0 and ((t == 0) or (t != 0 )):
						pred.append(self.alphabet[int(p[t])])
				dec_preds.append(''.join(pred))

		return dec_preds



def lex_free_acc(preds,labels,converter,loss,converter_gt =None,HTR = False):
	"""
	preds: output from the network, [B,max_len,nClass]
	labels: groundtruth. list of index in length [B*max_len]
	loss: ce or ctc
	output: acc and two strings
	ind: which sample in the batch you want to decode
	"""
	sen_correct,cer,total_char,total_wrds,wer = 0.,0.,0.,0.,0.
	cers = []
	pred_strs,gt_strs = [],[]
	if len(preds.shape)>=3:
		preds = torch.argmax(preds,dim =-1)
	preds_size = torch.IntTensor([preds.size(1)] * preds.size(0))
	if converter_gt is None:
		converter_gt = converter

	if loss == 'ctc':
		# preds includes break tokens for ctc 
		text_preds = converter.decode_CTC(preds,'pred')
		labels = labels.view(preds.size(0),-1)
		text_labels = converter_gt.decode_CTC(labels,'label')

	elif loss =='ce':
		preds_flat = preds.contiguous().view(-1)
		text_preds = converter.decode(preds_flat.data, preds_size, raw=False)
		labels = labels.view(preds.size(0),-1)
		if HTR:
			text_labels = converter.decode(labels.contiguous().view(-1).data, preds_size, raw=False)
		else:
			text_labels = converter_gt.decode_CTC(labels,'label')

	if isinstance(text_preds, str):
		text_preds = [text_preds]
	if isinstance(text_labels, str):
		text_labels = [text_labels]
	for pred, target in zip(text_preds,text_labels):
		pred = pred.lower()
		target = target.lower()

		# post-processing #
		while '  ' in pred:
			pred = pred.replace('  ',' ') 
		while '  ' in target:
			target = target.replace('  ',' ') 
		pred = pred.strip()
		target = target.strip()

		cer += editdistance.eval(pred,target)
		cers.append(editdistance.eval(pred,target)/len(target))

		pred_wrds = pred.split()
		tar_wrds = target.split()
		wer += editdistance.eval(pred_wrds, tar_wrds)

		if pred == target:
			sen_correct += 1
		pred_strs.append(pred)
		gt_strs.append(target)
		total_char += len(target)
		total_wrds += len(tar_wrds)

	sen_acc = sen_correct/len(text_preds)
	cer_p = cer / total_char
	wer = wer/ total_wrds

	return sen_acc,cer_p,wer,pred_strs,gt_strs




def convert_ctc_format(out,labels,lengths,alphabet):
	# forward : [batch,seq_len,dim] --> [seq_len,batch,dim] and apply log_softmax
	# labels: one dimensional tensor, batch*max_len --> various_len
	# act_lens: [batch],seq_len of each sample, before image input are padded
	# label_lens: [batch], real label_len of each sample
	# all in type LongTensor k
	batch_size,max_len = out.shape[:2]
	out = F.log_softmax(out,dim=-1)
	out = torch.transpose(out,0,1) 

	labelss = []
	new_labels,label_lens = [],[]
	for b_ind in range(labels.shape[0]):
		label = labels[b_ind].long()
		if len(alphabet)-1 not in label:
			import ipdb;ipdb.set_trace()
		EOS_ind = (label == len(alphabet)-1).nonzero()[0]
		labelss.append(label[:EOS_ind])
		label_lens.append(torch.IntTensor([len(label[:EOS_ind])]))
	label_lens = torch.cat(label_lens,dim=0).long()
	input_lens = torch.full((batch_size,), max_len, dtype=torch.long)
	labels = torch.cat(labelss,dim=0)
	return out, labels ,input_lens,label_lens




class min_entropy(nn.Module):
	def __init__(self):
		super(min_entropy, self).__init__()
		
	def forward(self,logits_flat,batch_size):
		# log_probs_flat: (batch * max_len, num_classes)
		probs_flat = nn.functional.softmax(logits_flat,dim=1)
		# logsumexp(x): (batch * max_len, num_classes)
		lse = torch.log(torch.sum(torch.exp(logits_flat),dim =1,keepdim=True))
		# dot product between [softmax(x),x - logsumexp(x) ] 
		entropy_probs_flat = torch.bmm(probs_flat.unsqueeze(-2),(logits_flat-lse).unsqueeze(-1))
		entropy = -entropy_probs_flat.squeeze(-1).squeeze(-1).sum() / batch_size

		return entropy 



def make_binary_map(char_seg_labels,seg_labels,img_lengths,constant =-1):

	sub = torch.stack([char_seg_labels-1]*char_seg_labels.shape[-1],dim=-1).to(torch.float32) -torch.stack([seg_labels]*char_seg_labels.shape[-1],dim=1).to(torch.float32)-1.
	zeros = torch.zeros(sub.shape).cuda()
	zeros = zeros.fill_(constant)
	binary_map = torch.eq(sub,zeros).to(torch.float32).transpose(1,2) # [B,W,H]
	max_len = max(img_lengths[::2]).item()

	binary_map = binary_map[:,:max_len,:].contiguous().view(-1,zeros.shape[-1])
	left_b = max_len*np.arange(0,char_seg_labels.shape[0])
	right_b = left_b + img_lengths[::2].numpy()
	valid_indices = [] 
	for i,left in enumerate(left_b):
		valid_indices.append(torch.arange(left,right_b[i]))
	valid_indices = torch.cat(valid_indices,dim=0).long().cuda()

	return binary_map,valid_indices


def generate_bound_labels(seg_labels):
	bound_labels = []
	for b_ind in range(seg_labels.shape[0]):
		bound_labels.append(torch.cat([label_bound(seg_labels[b_ind]),torch.zeros(1)]))
	bound_labels =pad_sequence(bound_labels,batch_first=True)
	return bound_labels #[B,W]


def count_rep(x):
	x  = x.cpu().numpy()
	res = [0]
	for idx, (a, b) in enumerate(zip(x[:-1], x[1:])):
		if a == b:
			tmp = res[idx] + 1
			res.append(tmp)
		else:
			res.append(0)
	res = torch.FloatTensor(res)
	return res


def label_bound(x):
	# 1 for boundary, 0 for non-boundary 
	x  = x.cpu().numpy()
	res = []
	for idx, (a, b) in enumerate(zip(x[:-1], x[1:])):
		if a == b:
			res.append(0)
		else:
			res.append(1)
	res = torch.FloatTensor(res)
	return res


def count_rep2(x):
	x  = x.cpu().numpy()
	res = []
	ccc=1
	for idx, (a, b) in enumerate(zip(x[:-1], x[1:])):
		if a == b :
			ccc+= 1
			if idx == len(x)-2:
				res.append(ccc)
		else:
			res.append(ccc)
			ccc=1
	res = torch.FloatTensor(res)
	return res


def compute_logits(x,char_seg_labels,num_classes):

	# x [B,1,W,H]
	# generate glyph-width map M
	onehot = torch.zeros(x.shape[0],num_classes,x.shape[-1]).cuda() #[B,CLASS,H]
	ones = torch.ones(char_seg_labels.unsqueeze(1).shape).cuda()
	onehot.scatter_(1, char_seg_labels.unsqueeze(1).long(), ones)
	label_map = torch.stack([onehot]*x.shape[2],dim=2) #[B,class,W,H]

	# compute class predictions
	logits = (x.expand_as(label_map))* label_map
	nonzeros = x.shape[-1] - (logits == 0).sum(dim=-1) 
	ones = torch.ones(nonzeros.shape).cuda().to(torch.int64)
	nonzeros = torch.where(nonzeros <ones,ones,nonzeros)
	logits = logits.sum(-1)/nonzeros.to(torch.float32)
	logits = logits.transpose(1,2)

	return  logits.reshape(-1, num_classes)



def crop_patches(imgs,seg,counter,volume):

	# seg in shape[1,w//2]
	for ind in range(bound_locs.shape[-1]-1):
		left = bound_locs[ind]+2
		right = bound_locs[ind+1]+2
		if right -left <4 or right-left>18:
			continue
		patch = img[:,left:right]
		torchvision.utils.save_image(patch.to(torch.uint8),osp.join(folder,str(counter)+'.png'))
		counter +=1

	return counter


