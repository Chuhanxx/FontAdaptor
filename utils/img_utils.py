import os
import glob
import torch
import torchvision
import numpy as np
import os.path as osp

import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import ImageFilter
from PIL import Image, ImageFile, ImageEnhance,ImageDraw

def gradient_bg(im,alpha):

	gradient = Image.new('L', im.size, color=0)
	draw = ImageDraw.Draw(gradient)

	f_co = np.random.random()*255
	t_co = 255-f_co
	for i, color in enumerate(interpolate(f_co, t_co, im.width * 2)):
		draw.line([(i, 0), (0, i)], fill=color, width=1)

	im_composite = Image.blend(im,gradient,alpha)

	return im_composite


def add_gradient(line_img):

	bg_im = ImagePatch()
	line_img =gradient_bg(line_img,np.random.random()*0.6)
	bg = bg_im.sample(line_img.size)
	bg = bg.convert('L')
	bbb = torch.tensor(np.array(line_img).astype(np.float32),dtype=torch.float32).transpose(0,1)
	line_img = Image.blend(bg,line_img,1-np.random.random()*0.5)

	return line_img


def add_interference(line_img,bg):

	# tensor input, [1,H,W]
	ori_h = line_img.shape[0]
	top =(torch.mean(line_img,1)!=bg).nonzero()[0]
	bottom = (torch.mean(line_img,1)!=bg).nonzero()[-1]
	margin =  ori_h - bottom +top
	h1 = int(np.ceil(margin/4))
	h2 = ori_h -h1
	nomargin = line_img[top:bottom,:]
	noise_h =ori_h - h2 + h1

	if int(bottom - top) < 2.5*noise_h:
		return line_img
	else:
		line_img = line_img[h1:h2,:]
		if np.random.random() <0.5:
			noise = flip(nomargin[-noise_h:,:],-1)
			line_img = torch.cat([noise,line_img],dim=0)
		else:
			noise = flip(nomargin[:noise_h,:],-1)
			line_img = torch.cat([line_img,noise],dim=0)

		return line_img


def flip(line_img,dim):

	idx = [i for i in range(line_img.size(dim)-1, -1, -1)]
	idx = torch.LongTensor(idx)

	return line_img.index_select(dim, idx)



def interpolate(f_co, t_co, interval):

	det_co =(t_co - f_co) / interval    

	return [round(f_co + det_co * i) for i in range(interval)]



def add_show_through(line_img,line_flip):
	# line_img torch tensor [W,H]

	line_img = Image.blend(line_flip,line_img,1-np.random.random()*0.3)
	line_img = functional.adjust_contrast(line_img, 1+np.random.random()*1.5)
	return line_img



def locate_margin(img,axis='h',bg = 185):
	# locate vertical margin
	# input img in PIL format

	im = np.array(functional.to_grayscale(img))
	if np.max(im) != bg:
		im [im>150] =bg

	if axis == 'h':
		nonbg= (np.mean(im,1)!=bg).nonzero()[0] # compute mean horizontally 
	elif axis == 'w':
		nonbg= (np.mean(im,0)!=bg).nonzero()[0]

	if len(nonbg) >2:
		top = nonbg[0] # or left
		bottom = nonbg[-1] # or right
	else: 
		top = 0
		if axis == 'h':
			bottom = img.size[1]
		else:
			bottom = img.size[0]

	return top,bottom
	


def random_crop(top,bottom,left,right,img):

	w,h = img.size
	left = np.max([0,left-2])
	top =np.max([0,top-2])
	right = np.min([w,right+2])
	bottom = np.min([h,bottom+2])
	img =  img.crop((left, top, right, bottom))
	w,h =img.size
	if w<1 or h <1:
		print(left, top, right, bottom)

	return img



def random_pad(img,w_max=2,h_max=23):

	#pad tuple (left, top, right, bottom) or (w,h)
	padding = (np.random.randint(0,w_max),np.random.randint(3,h_max),np.random.randint(0,w_max),np.random.randint(3,h_max))
	pad = transforms.Pad(padding, padding_mode='edge')
	try:
		img = pad(img)
	except KeyError:
		raise TypeError(img.size)

	return img



def resize_im_fixed_h(img,height):

	# input in PIL
	w,h = img.size
	new_w = int(height*w/h)
	new_w = np.max([1,new_w])

	return img.resize((new_w, height),resample=Image.BILINEAR)



def resize_im_fixed_w(img,width,fixed_h):

	# input in PIL
	w,h = img.size
	new_h = int(width*h/w)
	img = img.resize((width, new_h),resample=Image.BILINEAR)
	padding = (0, (fixed_h - new_h)//2,0,fixed_h - new_h- (fixed_h - new_h)//2) # left, top, right, bottom  
	pad = transforms.Pad(padding, padding_mode='edge')
	img = pad(img)

	return img



def blur_affine(img):

	img =  img.filter(ImageFilter.BLUR(1))
	shear = transforms.RandomAffine(shear =0.2)

	return(sheat(img))



def down_sample(imgs_c,imgs_l,ws_c,ws_ori_c):

	down_ratio =  720/sum(ws_ori_c)

	for ind,img_c in enumerate(imgs_c): 
		ori = img_c.shape[-1]
		img_c = downsample_img(img_c,down=(1,down_ratio))
		real_down_ratio = img_c.shape[-1]/ori
		ws_c[ind*3:ind*3+3] = [int(w*real_down_ratio) for w in ws_c[ind*3:ind*3+3] ]
		imgs_c[ind] = img_c
		img_l = downsample_img(imgs_l[ind],down=(1,down_ratio))
		imgs_l[ind] = img_l

	return imgs_c,imgs_l,ws_c



def downsample_img (img_c,down=(1,1)):

	ori_shape =  len(img_c.shape)
	while  len(img_c.shape) !=4:
		img_c = img_c.unsqueeze(0)
	img_c = F.interpolate(img_c,scale_factor =down,mode = 'bilinear')
	while len(img_c.shape) !=ori_shape:
		img_c = img_c.squeeze(0)

	return img_c



def layout_imgs(imgs,ws,bg,blank,fg=55):
	# paste imgs onto one canvas with random margin
	# but avoid collusion of characters at the same time 
	# skip the image of blank space

	if len(ws) > 2*len(imgs):
		flag = True
		ws =[0]+ [w for w in ws if w!=0]
	else:
		flag =False
	margin =np.random.randint(-2,np.max(ws)//2)
	if np.sum(ws)-len(imgs)*margin >720:
		margin =np.random.randint(-1,np.max(ws)//2)
	bounds =  np.cumsum(ws).tolist()
	canvas = torch.zeros([imgs[0].shape[0],np.sum(ws)+100],dtype =torch.float32).fill_(bg)
	ws_new= [0]
	rights = [0]
	real_margins = [0]

	for b_i,b in enumerate(bounds):
		if b_i == len(bounds)-1:break
		if b_i ==1 or b_i ==0: 
			b = rights[-1]
		else:
			b = np.max([0,rights[-1]-np.min([margin,imgs[b_i].shape[1]-1])])
		if canvas[:,b:b+ws[b_i+1]].shape !=imgs[b_i].shape:
			import ipdb;ipdb.set_trace()
		collusion = binarize(canvas[:,b:b+ws[b_i+1]],imgs[b_i],bg,fg=fg)
		while torch.sum(collusion)>0:
			b += 1
			if canvas[:,b:b+ws[b_i+1]].shape !=imgs[b_i].shape:
				import ipdb;ipdb.set_trace()
			collusion = binarize(canvas[:,b:b+ws[b_i+1]],imgs[b_i],bg,fg=fg)

		canvas[:,b:b+ws[b_i+1]] =torch.min(canvas[:,b:b+ws[b_i+1]],imgs[b_i])

		real_margins.append(rights[-1]-b)
		rights.append(b+ws[b_i+1])
	canvas = canvas[:,:rights[-1]]
	real_margins.append(0)
	for ind,w in enumerate(ws[1:]):
		if not flag:
			ws_new.append(w-np.floor(real_margins[ind+1]/2)-np.ceil(real_margins[ind+2]/2))
		else:
			ws_new.extend((0,w-np.floor(real_margins[ind+1]/2)-np.ceil(real_margins[ind+2]/2),0))

	return canvas,ws_new




def binarize(img1,img2,bg,fg):
	imga = img1.clone()
	imgb = img2.clone()
	threshold = (bg+fg)//2
	imga[img1<threshold] =1
	imgb[img2<threshold] =1
	imga[img1>threshold] =0
	imgb[img2>threshold] =0 

	collusion = torch.sum(imga*imgb,dim=0)
	return collusion



class ImagePatch(object):

	def __init__(self, dir, min_px=7):
		"""
		IMDIRS: list of paths to directories from
				which image patches can be extracted
				for use as background / other image layers.
		"""
		# minimum number of pixels in the sampled image:
		self.min_px = min_px
		# max no. of trials before failing:
		self.max_samp_iter = 10
		# parameters based on SVT data distribution:
		self.p_beta = [2.0,8.0] 
		self.imdirs = [osp.join(dir,'bg_ims')]
		ims = []
		for d in self.imdirs:
			ims += [osp.join(d,f) for f in os.listdir(d) if osp.isfile(osp.join(d,f))]
		self.ims = ims

	def extract_patch(self,im,hw):
		"""
		Returns a patch of size = HW = [H,W] from the image IM.
		"""
		im,hw = np.array(im), np.array(hw)
		im_hw = np.array(im.shape[:2])
		assert(np.all(im_hw > self.min_px))
		# get the aspect ratio of the patch to sample:
		asp = hw / hw[0]
		# get the valid range for this dimension:
		max_d = np.argmax(asp)
		min_d = np.mod(max_d+1,asp.size)
		min_sz = asp[max_d] * self.min_px
		max_sz = im_hw[max_d]
		# sample the size of the biggest side:
		rand_num = np.random.beta(self.p_beta[0],self.p_beta[1])
		sz = min_sz + (max_sz-min_sz) * rand_num
		# get the size of the patch to sample:
		sample_sz = np.zeros_like(im_hw)
		sample_sz[max_d] = int(np.round(sz))
		div_asp = asp[max_d] if max_d==1 else 1.0/asp[min_d]
		sample_sz[min_d] = int(np.round(sz/div_asp))
		# now sample the position of the patch in the image:
		p_h = np.random.randint(0,im_hw[0]-sample_sz[0]+1)
		p_w = np.random.randint(0,im_hw[1]-sample_sz[1]+1)
		# sample the patch:
		im_patch = im[p_h:p_h+sample_sz[0],p_w:p_w+sample_sz[1],...]
		# resize the patch:
		im_patch = Image.fromarray(im_patch)
		return im_patch.resize(hw[::-1],resample=Image.BICUBIC)

	def sample(self, hw):
		"""
		Returns an image patch of size = hw = [HEIGHT,WIDTH].
		"""
		iter = 0
		hw = (hw[1],hw[0])
		while iter < self.max_samp_iter:
			try:
				iter += 1
				im = self.ims[np.random.randint(0,len(self.ims))]
				im = Image.open(im)
				im_patch = self.extract_patch(im,hw)
				return im_patch
			except:
				continue
		return np.zeros((hw[0],hw[1],3),dtype=np.uint8)


