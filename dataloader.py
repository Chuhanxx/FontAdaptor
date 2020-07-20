import os
import sys
import cv2
import random
import torch
import glob
import numpy as np
import scipy
import _pickle as cp
import string
import torchvision

from utils.utils import *
from utils.img_utils import *
from numpy.random import *

import imgaug as ia
import os.path as osp

import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

from os.path import join
from imgaug import augmenters as iaa
from scipy.misc import imread, imresize
from torch.utils.data import Dataset,Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,Sampler
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import list_dir
from PIL import Image, ImageFile, ImageFilter,ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestLoader(data.Dataset):
	def __init__(self, args, aug=False):
		self.args = args
		self.root =  args.root
		self.ims_dir = join(self.root,'ims')
		self.file = join(self.root ,'gt/test/test_'+self.args.fontname+'.txt')
		self.subdir = 'test_new'
		self.char_folder =  'chars'

		self.cut =3
		self.get_all_samples()

	def __len__(self):

		return len(self.volumes)

	def get_all_samples(self):

		self.gt_file = open(self.file,'rb').readlines()
		self.volumes, self.imgs, self.labels= self.parse_samples(self.gt_file)


	def parse_samples(self,img_list):

		volumes,imgs,labels,seg_labels= [],[],[],[]
		for ind,line in enumerate(img_list):
			line = line.decode()
			parts = line.strip().split('\t')
			volumes.append(parts[0])
			imgs.append(join(self.ims_dir,parts[0],self.subdir,parts[1]))
			labels.append(parts[2])
		return volumes,imgs,labels

	def __getitem__(self, index):

		img = Image.open(self.imgs[index])
		w,h = img.size
		new_w = int(self.args.load_height*w/h)
		img= img.resize((new_w, self.args.load_height),resample=Image.BILINEAR)
		lengths = []

		label = self.labels[index]
		converter = strLabelConverter(self.args.alphabet_gt)
		text, _ = converter.encode(label) # text to list of inds
		volume = self.volumes[index]

		img = functional.to_grayscale(img)
		img = np.array(img).astype(np.float32)
		img = np.expand_dims(img, axis=2)
		line_img = torch.from_numpy(img.transpose((2, 1, 0)))
		line_img = line_img[0,:]

		self.alphabet = self.args.alphabet
		self.chars,char_inds,classes,counter  = [],[],[],1
		self.char_base =  join(self.ims_dir,self.args.fontname)
		for char in self.args.alphabet[1:-1]:
			if self.args.cross:
				self.chars.append(join(self.ims_dir,self.args.ref_font,'chars',char+'.png'))
			else:
				self.chars.append(join(self.char_base,self.char_folder,char+'.png'))
		char_inds = np.arange(1,28).tolist()


		# load reference images
		imgs,ws,masks = [],[0],[]

		for n in range(len(self.chars)):
			img = Image.open(self.chars[n])
			w,h = img.size
			new_w = int(self.args.load_height*w/h)
			img= img.resize((new_w, self.args.load_height),resample=Image.BILINEAR)
			img = functional.to_grayscale(img)
			img = np.array(img).astype(np.float32)
			if n !=0 and  new_w >10:
				left =self.cut
				right = new_w - self.cut
				img = img[:,left:right]

			ws.append(img.shape[1])
			img = np.expand_dims(img, axis=2)
			imgs.append(img) #[H,W,C]

		char_img =np.concatenate(imgs,axis=1)[:,:,0]
		if int((char_img.shape[1]/2)+0.5) >=self.args.d_model:
			old_w = char_img.shape[1]
			char_img =  resize_im_fixed_w(Image.fromarray(char_img),self.args.d_model*2,self.args.load_height)
			char_img = np.array(char_img).astype(np.float32)
			ratio = (2*self.args.d_model-10)/old_w
			ws = [w*ratio for w in ws]
		char_img = torch.from_numpy(char_img.transpose((1, 0))) #[H,W,C] --> [W,H]

		if np.mean(ws) <=8*(self.args.load_height/32) and char_img.shape[0]<= self.args.d_model and line_img.shape[0]<self.args.d_model :
			ws = [w*2 for w in ws]
			char_img = char_img.repeat_interleave(2,0)
			line_img = line_img.repeat_interleave(2,0)

		lengths.append(int((char_img.shape[0]/2)+0.5))
		char_img = F.pad(char_img,(0,0,0,self.args.d_model*2-char_img.shape[0]),'constant',0)
		char_len = int((char_img.shape[0]/2)+0.5)
		w_ratios = np.cumsum(ws)/char_img.shape[0]
		char_seg_label =torch.zeros(self.args.d_model).fill_(len(self.alphabet)-1) #[27,W/2]
		for w_ind in range(len(w_ratios)-1):
			w_ratio1 = w_ratios[w_ind] 
			w_ratio2 = w_ratios[w_ind+1] 
			char_seg_label[int(w_ratio1*char_len):int(w_ratio2*char_len)] = float(char_inds[w_ind])
	

		lengths.append(int((line_img.shape[0]/2)+0.5))		
		if self.args.d_model*2>=line_img.shape[0]:
			line_img = F.pad(line_img,(0,0,0,self.args.d_model*2-line_img.shape[0]),'constant',0)

		im_len = int((line_img.shape[0]/2)+0.5)

		length = torch.IntTensor(lengths[::-1])
		text = torch.IntTensor((text + [len(self.args.alphabet_gt)-1] * lengths[0])[:lengths[0]])

		imgs = pad_sequence([line_img, char_img],batch_first=True)
		return imgs, char_seg_label, text, length,self.alphabet



def text_collate_eval(batch):	
	#  imgs, char_seg_label, text, length,self.alphabet
	imgs,labels,seg_labels,char_seg_labels,char_seg_labels2,img_lengths = [],[],[],[],[],[]

	for sample in batch:
		img_lengths.append(sample[0][0].shape[0])
	batch_size = len(batch)
	sorted_idx = np.argsort(img_lengths)[::-1]
	batch = [batch[i] for i in sorted_idx]
	lengths,inds = [],[]

	for ind,sample in enumerate(batch):
		#[line_img, char_img], char_seg_label, seg_label, text
		imgs.append(sample[0][0]) 
		imgs.append(sample[0][1])
		lengths.append(sample[3][0])
		lengths.append(sample[3][1])
		char_seg_labels.append(sample[1]) 
		labels.append(sample[2])
		if sample[-1] ==1: inds.append(ind)
		alphabet = sample[-1]

	img_lengths = torch.IntTensor([i for i in lengths])
	imgs =  pad_sequence(imgs,batch_first=True,padding_value=0)
	char_seg_labels = torch.stack(char_seg_labels,dim=0)
	labels = pad_sequence(labels,batch_first=True,padding_value=28).view(-1)
	return imgs, char_seg_labels,labels,img_lengths,alphabet

def baseline_collate(batch):	

	imgs = []
	labels = []
	img_lengths = []
	for sample in batch:
		img_lengths.append(sample[0].shape[0])
	sorted_idx = np.argsort(img_lengths)[::-1]
	batch = [batch[i] for i in sorted_idx]

	for sample in batch:
		imgs.append(sample[0]) # list of [W,H]s
		labels.append(sample[1]) 

	img_lengths = torch.IntTensor([int((img_lengths[i]/2)+0.5) for i in sorted_idx])
	imgs = pad_sequence(imgs,batch_first=True)
	labels = pad_sequence(labels,batch_first=True,padding_value=28).view(-1)

	return imgs, labels,img_lengths




class Omniglot(VisionDataset):
	"""`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
	Args:
		root (string): Root directory of dataset where directory
			``omniglot-py`` exists.
		background (bool, optional): If True, creates dataset from the "background" set, otherwise
			creates from the "evaluation" set. This terminology is defined by the authors.
	"""
	folder = 'omniglot-py'

 
	def __init__(self, args,root = '',alpha_ind=None, 
					background=True,selected_chars = None,size=15000):
		super(Omniglot, self).__init__(root)
		self.background = background # True for training, False for testing 
		self.args = args
		self.lines = 500
		self.root = root
		self.target_folder = join(self.root, self.folder,self._get_target_folder())

		self._alphabets = list_dir(self.target_folder)

		if not self.background:
			self._alphabets = [self._alphabets[alpha_ind]]
		self._characters = [[c for c in list_dir(join(self.target_folder, a))] for a in self._alphabets]


		self.text_list = open(os.path.join(self.root , 'gt', 'train/train_regular+bold+light+italic_50_resample.txt')).readlines()
		self.txts = self.extract_txt()
		self.color_jitter =transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)
		self.ToPIL =  transforms.ToPILImage()
		self.bg_im = ImagePatch(join(self.root,'ims'))
		self.char_aug_threshold = 0.5
		self.len = size
		self.alpha_ind = alpha_ind


	def __len__(self):
		if self.background:
			return len(self._alphabets)*self.len
		else:
			return self.len

	def extract_txt(self):
		txts= []
		for ind,line in enumerate(self.text_list):
			parts = line.strip().split('\t')
			txts.append(parts[2])
		return txts

	def __getitem__(self, index):
		"""
		Args:
			index (int): choose which alphabet
		Returns:
			tuple: (image, target) where target is index of the target character class.
		"""

		if self.background :
			index = index//self.len
		else:
			index =0

		self.bg = np.random.randint(120,200)
		self.fg = np.random.randint(20,70)

		imgs_c,imgs_l,lengths,ws_c,ws_l,ws_ori_c = [],[],[],[0],[0],[0]
		txt_no = np.random.randint(0,len(self.txts))
		# random width for blank spaces 
		blank_width = np.random.choice(range(6,13))
		blank = torch.tensor(np.zeros([1,32,blank_width]),dtype=torch.float32)
		blank = blank.fill_(self.bg)
		# margin before and after sentences
		margin =  torch.tensor(np.zeros([1,32,4]),dtype=torch.float32)
		margin = margin.fill_(self.bg)

		if np.random.random()>0 or not self.background:
			im_paths_char,im_paths_line=self.select_glyphs(index)
			max_char_no = np.min([26,len(self._characters[index])])
		else:
			im_paths_char,im_paths_line=self.select_chars(index)
			max_char_no = len(im_paths_char)

			
		# choose transformartion params for this line 
		if np.random.random()>0.5 and self.background and max_char_no!= 20:
			self.rotations = np.random.choice(range(-180,180),size = max_char_no,replace=True)
		else: 
			self.rotations = [0]*max_char_no
		if np.random.random()>0.5 and self.background:
			self.shears = np.random.choice(range(-5,5),size = max_char_no,replace=True)
			self.dilation_k = np.random.choice([1,3,5])
		else: 
			self.shears = [0]*max_char_no
			self.dilation_k = 1
		if self.background:
			self.down_ratio = 0.9 - np.random.random()*0.2
		else:
			self.down_ratio = 1

		imgs_l.append(blank)
		imgs_c.append(blank.squeeze(0))
		ws_c.extend((0,blank.shape[-1],0))
		ws_ori_c.extend((0,blank.shape[-1],0))
		char_prob = np.random.random()
		if char_prob>self.char_aug_threshold and self.background:
			self.char_aug_flag = True
		else:
			self.char_aug_flag = False

		for idx in range(len(im_paths_char)):
			# remove the horizontal margin
			if not self.char_aug_flag or not self.background:
				img_c =self.load_img(im_paths_char[idx],blank_width,idx)
				ws_c.extend((0,img_c.shape[-1],0))
				ws_ori_c.extend((0,img_c.shape[-1],0))
			else: 
				img_c,ws_c,ws_ori_c =self.load_aug_img(im_paths_char[idx],ws_c,ws_ori_c,idx)
			imgs_c.append(img_c[0])
			img_l = self.load_img(im_paths_line[idx],blank_width,idx)
			imgs_l.append(img_l)

		if sum(ws_ori_c) > 720:
			imgs_c,imgs_l,ws_c = down_sample(imgs_c,imgs_l,ws_c,ws_ori_c)

		char_line_img = torch.cat(imgs_c,dim=-1)
		char_line_img = torch.transpose(char_line_img,0,1) #[H,line_w] --> [W,H]

		converter = strLabelConverter(self.args.alphabet)
		text,_ = converter.encode(self.txts[txt_no])
		
		line_parts = [margin]
		ws_l.append(margin.shape[-1])
		text_new = [1]
		for k,char_i in enumerate(text):
			char_i = char_i-1
			if k!=0 and char_i ==0 and text_new[-1]==1:continue
			if char_i > max_char_no:continue
			text_new.append(char_i+1)
			if char_i ==0 and np.random.random()>0.5:
				blank_var = - np.random.randint(0.5*blank_width+1)
			else:
				blank_var = 0 
			ws_l.append(imgs_l[char_i].shape[-1]+blank_var)
			line_parts.append(imgs_l[char_i][:,:,:imgs_l[char_i].shape[-1]+blank_var])
		line_parts.append(margin)
		ws_l.append(margin.shape[-1])
		text_new.append(1)

		if np.random.random() >0.3 and self.background:
			line_img = torch.cat(line_parts,dim=-1)#[H,line_w]
			line_img =downsample_img(line_img,down =(self.down_ratio,self.down_ratio)).squeeze(0)
			pad1 = (self.args.load_height -line_img.shape[0])//2
			padding = (0,0,pad1,self.args.load_height-pad1-line_img.shape[0]) #[left,right,top,bottom]
			line_img = torch.nn.functional.pad(line_img,padding,mode='constant',value=self.bg)
			line_img = add_interference(line_img,self.bg)
			line_img = torch.transpose(line_img,0,1)
			ws_l = [w*self.down_ratio for w in ws_l]
		else:
			line_img = torch.transpose(torch.cat(line_parts,dim=-1)[0],0,1)
		   
		if self.background :
			line = self.ToPIL(line_img/255)
			line_flip = self.ToPIL(torch.flip(line_img,[0,1])/255)
			line_img =gradient_bg(line,np.random.random()*0.2)
			bg = self.bg_im.sample(line_img.size)
			bg = bg.convert('L')
			line_img = Image.blend(line_flip,line_img,1-np.random.random()*0.18)
			blend = Image.blend(bg,line_img,1-np.random.random()*0.5)
			line_img =transforms.functional.adjust_contrast(blend, 1+np.random.random()*1.5)
			if np.random.random()>0.5:
				line_img = ImageOps.invert(line_img)
			line_img = torch.tensor(np.array(line_img).astype(np.float32),dtype=torch.float32)

		lengths.append(line_img.shape[0])
		lengths.append(char_line_img.shape[0])

		char_inds = np.arange(1,28).tolist()
		char_seg_labels,char_seg_label2 = self.make_char_seg_labels(ws_c,char_inds,char_line_img)
		char_line_img = F.pad(char_line_img,(0,0,0,self.args.d_model*2-char_line_img.shape[0]),'constant',0)
		lengths = torch.IntTensor([int((l/2)+0.5) for l in lengths])
		text = torch.IntTensor((text_new + [len(self.args.alphabet)-1] * int((lengths[0]/2)+0.5))[:int((lengths[0]/2)+0.5)])

		if self.background: #testing 
			seg_labels = self.make_seg_labels(ws_l,text_new,line_img)-1
			return  [line_img,char_line_img],char_seg_labels,text,seg_labels.double(),lengths,char_seg_label2,0

		else:
			imgs = pad_sequence([line_img, char_line_img],batch_first=True)
			return imgs, char_seg_labels, text, lengths,self.args.alphabet,0
   


	def make_seg_labels(self,ws,char_inds,img):
		w_ratios = np.cumsum(ws)/img.shape[0]
		im_len = int((img.shape[0]/2)+0.5)

		seg_label =torch.zeros(np.max([self.args.d_model,im_len])).fill_(len(self.args.alphabet)-1) #[27,W/2]
		for w_ind in range(len(w_ratios)-1):
			w_ratio1 = w_ratios[w_ind] 
			w_ratio2 = w_ratios[w_ind+1] 
			seg_label[int(w_ratio1*im_len):int(w_ratio2*im_len)] = float(char_inds[w_ind])

		return seg_label
	
	def make_char_seg_labels(self,ws,char_inds,char_img):
		w_ratios = np.cumsum(ws)/char_img.shape[0]
		char_len = int((char_img.shape[0]/2)+0.5)
		char_seg_label =torch.zeros(self.args.d_model).fill_(len(self.args.alphabet)-1) #[27,W/2]
		char_seg_label2 =torch.zeros(self.args.d_model).fill_(len(self.args.alphabet)-1)  # for cheat sheet, without extra blank around glyphs
		for i in range(len(w_ratios)//3):
			bound1 = w_ratios[3*i] 
			start = w_ratios[3*i+1]
			end = w_ratios[3*i+2]
			bound2 =  w_ratios[3*i+3]
			char_seg_label[int(bound1*char_len):int(start*char_len)] = float(char_inds[0])
			char_seg_label[int(start*char_len):int(end*char_len)] = float(char_inds[i])
			char_seg_label[int(end*char_len):int(bound2*char_len)] = float(char_inds[0])
			char_seg_label2[int(bound1*char_len):int(bound2*char_len)] = float(char_inds[i])
		return char_seg_label,char_seg_label2

	def _get_target_folder(self):
		return 'images_background' if self.background else 'images_evaluation'


	def select_glyphs(self,index):
		# select glyphs from an alphabet
		im_paths_char,im_paths_line = [],[]
		alphabet = self._alphabets[index]

		characters = np.random.choice(self._characters[index],np.min([26,len(self._characters[index])]),replace=False)
		char_paths = [join(self.target_folder, alphabet,char) for char in characters]
		for char_path in char_paths:
			char_no_char = np.random.randint(1,len(glob.glob(join(char_path,'*.png')))+1)
			im_paths_char.append(glob.glob(join(char_path,'*_'+str(char_no_char).zfill(2)+'.png'))[0])
			if self.background or self.args.cross:
				char_no_line = np.random.randint(1,len(glob.glob(join(char_path,'*.png')))+1)
			else:
				char_no_line = char_no_char
			im_paths_line.append(glob.glob(join(char_path,'*_'+str(char_no_line).zfill(2)+'.png'))[0])
		return im_paths_char,im_paths_line

	def select_chars(self,index,random=True):
		# select chars from a glyph
		im_paths_char,im_paths_line = [],[]
		alphabet = self._alphabets[index]

		glyph= np.random.choice(self._characters[index],1,replace=False)
		char_paths = glob.glob(join(self.target_folder, alphabet,glyph[0],'*.png'))
		random.shuffle(char_paths)    
		return char_paths,char_paths 

	def load_img(self,path,blank_width,idx,aug=True):
		img = Image.open(path, mode='r').convert('L')
		img = img.resize(( self.args.load_height, self.args.load_height),resample=Image.BILINEAR)
		img = torchvision.transforms.functional.affine(img, self.rotations[idx],(0,0), 1, self.shears[idx], resample=0, fillcolor=255)
		if  self.rotations[idx] %3 ==0:
			img = torchvision.transforms.functional.hflip(img)
		if  self.rotations[idx] %5 ==0:
			img = torchvision.transforms.functional.vflip(img)
		img = img.filter(ImageFilter.MinFilter(self.dilation_k))
		img = torch.tensor(np.array(img).astype(np.float32)).unsqueeze(0)
		img[img>120] = self.bg
		img[img<120] = self.fg
		margin = np.random.randint(0,blank_width//2)
		if not self.background :
			margin = 4
		nonbg = (torch.mean(img[0],dim=-2)!=self.bg).nonzero()
		if len(nonbg) >2:
			left = int(nonbg[0])
			right = int(nonbg[-1])
			if right - left> -margin+4:
				img = img [:,:,np.max([0,left-margin//2]):np.min([right+margin//2,img.shape[-1]])] # [H,new_w] 
			elif right <= left:
				 img = img [:,:,np.max([0,left-1]):np.min([right+1,img.shape[-1]])]
			else: 
				img = img [:,:,np.max([0,left]):np.min([right,img.shape[-1]])]

		return img

	def load_aug_img(self,path,ws,ws_ori,idx):
		img = Image.open(path, mode='r').convert('L')
		img = img.resize(( self.args.load_height, self.args.load_height),resample=Image.BILINEAR)
		img = torchvision.transforms.functional.affine(img, self.rotations[idx],(0,0), 1, self.shears[idx], resample=0, fillcolor=255)
		if  self.rotations[idx] %3 ==0:
			img = torchvision.transforms.functional.hflip(img)
		if  self.rotations[idx] %5 ==0:
			img = torchvision.transforms.functional.vflip(img)

		img = img.filter(ImageFilter.MinFilter(self.dilation_k))
		img = np.array(img).astype(np.float32)
		img[img>120] = self.bg
		img[img<120] = self.fg
		img = Image.fromarray(np.uint8(img))
		top,bottom = locate_margin(img,bg = self.bg)
		left,right = locate_margin(img,axis='w',bg = self.bg)
		img = random_crop(top,bottom,left,right,img)
		img = random_pad(img,h_max=20)
		img= resize_im_fixed_h(img,self.args.load_height)
		left,right = locate_margin(img,axis='w')
		img = self.color_jitter(img)
		if np.random.random()<0.5 and self.background:
			img =  img.filter(ImageFilter.GaussianBlur(radius=0.6))
		img =  torch.tensor(np.array(functional.to_grayscale(img)).astype(np.float32)).unsqueeze(0)
		ws.extend((left,right-left,img.shape[-1]-right))
		ws_ori.extend((0,img.shape[-1],0))
		return img,ws,ws_ori

