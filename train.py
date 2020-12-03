# TODO: change iteration to epoch

import os
import argparse
import time
import torch
import matplotlib
import torchvision
matplotlib.use('Agg')

from dataloader import *
from utils.utils import *
from PIL import Image, ImageFile
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model.model import GlyphAdaptor

import _pickle as cp
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
cv2.setNumThreads(0)

def worker_init_fn(worker_id):                                                          
	np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
	parser = argparse.ArgumentParser(description='template_model')
	parser.add_argument('--name', default='init', type=str)

	## data setting 
	parser.add_argument('--root', default='./data',type=str)
	parser.add_argument('--trainset', default='attribute4', type=str,help = 'number of training attribute used')
	parser.add_argument('--evalset', default='FontSynth', type=str,help = 'FontSynth,100')
	parser.add_argument('--load_height', default=32, type=int)
	parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
	parser.add_argument('--lang', default="EN", type=str)
	parser.add_argument("--data",default = 'data',type=str,help="mixed,syn")
	parser.add_argument("--cross", dest="cross", action = 'store_true')
	parser.add_argument("--weight",default=1.0, type =float)

	# training params
	parser.add_argument("--gpus", dest="gpu", default="0", type=str)
	parser.add_argument("--char_aug", dest="char_aug", action = 'store_true')
	parser.add_argument("--downsample", dest="downsample", action = 'store_true')
	parser.add_argument("--init_ims",default=6, type =int)
	parser.add_argument('--d_model', default=360, type=int)
	parser.add_argument('--num_workers', default=12, type=int) 

	## model setting
	parser.add_argument('--alphabet', default='/ abcdefghijklmnopqrstuvwxyz-', type=str)
	parser.add_argument("--model",default='transformer',type=str)
	parser.add_argument('--resume_i', default=0, type=int)
	parser.add_argument("--TPS", dest="TPS", action = 'store_true')

	## optim setting
	parser.add_argument('--batch_size', default=24, type=int)
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--weight_decay', default=1e-5, type=float)
	parser.add_argument('--gamma', default=0.1, type=float)
	parser.add_argument('--optim', default='adam', type=str, help='sgd, adam, adadelta')
	parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
	parser.add_argument('--max_epoches', default=1000000, type=int)

	## output setting
	parser.add_argument('--log_iter', default=10, type=int)
	parser.add_argument('--eval_iter', default=2000,type=int )
	parser.add_argument('--save_iter', default=2000, type=int)
	parser.add_argument('--save_folder', default='./ckpts', type=str)
	parser.add_argument('--tbx_folder', default='./tbx', type=str)
	parser.add_argument('--max_iter', default=200000, type=int)
	
	
	args = parser.parse_args()

	make_dirs(args)
	writer = SummaryWriter(args.tbx_dir)

	args.nClasses = len(args.alphabet)
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.FloatTensor')

	## setup glyph converter
	converter = strLabelConverter(args.alphabet,)
	converter2 =strLabelConverter(args.alphabet[1:])
	args.alphabet_gt = args.alphabet

	## read list of training and validation fonts 
	fonts_list,val_fonts = load_fonts(args)
	args.baseline = False

	## setup model
	net = GlyphAdaptor(args)
	net = torch.nn.DataParallel(net).to(device)

	## load checkpoint 
	if args.resume_i !=0:
		resume_file = osp.join(args.save_folder,str(args.resume_i)+'.pth')
		checkpoint = torch.load(resume_file)
		net.load_state_dict(checkpoint['model_state_dict'],strict=True)

	## setup dataset
	if args.data == 'omniglot':
		trainset = Omniglot(args)
	elif args.data == 'eng':
		trainset =  LineLoader(args,converter,aug=True)
	elif args.data == 'omnieng':
		trainset1 =  LineLoader(args,converter,aug=True)
		trainset2 = Omniglot(args,args.root,size=len(trainset1)//45)
		trainset = torch.utils.data.ConcatDataset([trainset1,trainset2])
		print('english/synthetic dataset size ratio:',len(trainset1)/len(trainset2))
	else :
		print('no training dataset specified')
	train_loader = data.DataLoader(trainset, args.batch_size,num_workers=args.num_workers,
								  shuffle=True, collate_fn=text_collate, pin_memory=True, worker_init_fn=worker_init_fn)


	## setup optimizer
	parameters = net.parameters()
	if args.optim == 'sgd':
		optimizer = optim.SGD(parameters, lr=args.lr,
							  momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optim == 'adam':
		optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
	elif args.optim == 'adadelta':
		optimizer = optim.Adadelta(parameters,lr=1.0, rho=0.95, eps=1e-8)
		optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

	## train model
	cudnn.benchmark = True
	net.train()
	iter_counter = args.resume_i+1
	running_loss,running_loss1,running_loss2 = 0.,0.,0.

	for i in range(args.max_epoches):
		i = args.resume_i +i 
		t0 = time.time()
		for j, batch_samples in enumerate(train_loader):

			imgs, labels, char_seg_labels,img_lengths,seg_labels,char_seg_labels2= batch_samples
			imgs = torch.transpose(imgs.unsqueeze(1),2,3) #[B,C,H,W]

			char_seg_labels = char_seg_labels.to(device)
			preds, sims,sims_conv= net(imgs,char_seg_labels,img_lengths,char_seg_labels2) 

			loss1,loss2 = compute_loss(args,preds,img_lengths,sims,labels,\
							char_seg_labels,seg_labels)
			loss = loss1+args.weight*loss2

			optimizer.zero_grad()
			running_loss += loss.item()
			running_loss1+= loss1.item()
			running_loss2 += loss2.item()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_norm)
			optimizer.step()

			if iter_counter % (args.log_iter) == 0:
				t1 = time.time()
				writer.add_scalar('train/train_loss',running_loss/args.log_iter,iter_counter)
				writer.add_scalar('train/train_loss_ctc',running_loss2/args.log_iter,iter_counter)
				writer.add_scalar('train/train_loss_ce',running_loss1/args.log_iter,iter_counter)
				sen_acc,cer,wer,pred_samples,label_samples= lex_free_acc(preds,labels,converter,'ctc')

				print('iter:%6d  loss:%4.4f  cer:%4.1f  %4.1fs/batch'\
					%(iter_counter,running_loss/args.log_iter, cer*100, (t1-t0)/args.log_iter))
				writer.add_scalar('train/train_cer',cer,iter_counter)
				visual_txt1 = 'gt: '+ str(label_samples[-1])+ '----- logits_pred: '+str(pred_samples[-1])
				print(visual_txt1)

				t0 = time.time()
				running_loss,running_loss1,running_loss2 = 0.,0.,0.
				torch.cuda.empty_cache()


			if iter_counter % args.save_iter == 0:
				print('Saving state, epoch: %d iter:%d'%(i, j))
				torch.save({'model_state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict(),},\
					args.save_folder + '/'  + repr(iter_counter) + '.pth')
			

			if iter_counter % args.eval_iter == 0:
				## eval model

				net.eval()
				n_correct,cers,wers,counter = 0,0,0,0

				for val_font in val_fonts:
					args.fontname = val_font
					args.ref_font = val_font

					testset =  TestLoader(args,aug= False)

					seq_sampler = data.SequentialSampler(testset)
					test_loader = data.DataLoader(testset, 1, num_workers=args.num_workers,
												  sampler = seq_sampler,pin_memory=True, 
												  collate_fn=text_collate_eval,drop_last=False,
												  worker_init_fn=worker_init_fn)

					cer_font,acc_font,wer_font = [],[],[]

					for index, sample in enumerate(test_loader):
		
						imgs, char_seg_label, labels, length, alphabet= sample

						imgs = torch.transpose(imgs.unsqueeze(1),2,3)
						preds,sims,final_conv= net(imgs,char_seg_label,length.long(),char_seg_label) #[batch,len,classes]
						correct,cer,wer,pred_str,gt_str = lex_free_acc(preds,labels,converter,'ctc',converter_gt = converter)
						counter += 1
		
						cer_font.append(cer)
						acc_font.append(correct)
						wer_font.append(wer)
						n_correct += correct
						cers += cer
						wers += wer

					print('font:%s cer = %.1f wer = %.1f' %(args.fontname,np.mean(cer_font)*100,np.mean(wer_font)*100))

				acc = n_correct*1.0/counter
				char_er = cers*1.0 / counter
				word_er = wers*1.0 / counter
				print('accuracy=%.1f cer = %.1f wer = %.1f  total_samples= %d'\
					%(acc,char_er*100,word_er*100,counter))
				print('----------')
				if args.eval_iter != 1:
					writer.add_scalar('val/val_sentence_accuracy',acc,iter_counter)
					writer.add_scalar('val/val_cer',char_er,iter_counter)
					writer.add_scalar('val/val_wer',word_er,iter_counter)

			converter = strLabelConverter(args.alphabet)
			iter_counter += 1
			net.train()


def load_fonts(args):

	fonts_list = open(osp.join(args.root,'gt','train_att4.txt'),'r').readlines()
	val_fonts = []
	if args.evalset == 'FontSynth':
		lines = open(osp.join(args.root,'gt','test_FontSynth.txt'),'r').readlines()
	elif args.evalset == '100':
		lines = open(osp.join(args.root,'gt','val_100.txt'),'r').readlines()

	for line in lines:
		val_fonts.append(line.split('/')[-1].replace('.ttf','').replace('\n',''))

	return fonts_list,val_fonts

def make_dirs(args):

	if osp.exists(args.save_folder) == False:
		os.mkdir(args.save_folder)
	args.save_folder = osp.join(args.save_folder ,args.name)
	if osp.exists(args.save_folder) == False:
		os.mkdir(args.save_folder)

	args.tbx_dir =osp.join(args.tbx_folder,args.name)
	if osp.exists(args.tbx_folder) == False:
		os.mkdir(args.tbx_folder)

	if osp.exists(args.tbx_dir) == False:
		os.mkdir(args.tbx_dir)

	result_dir = osp.join(args.tbx_dir,'results')
	if osp.exists(result_dir) == False:
		os.mkdir(result_dir)


def compute_loss(args,preds,img_lengths,sims,labels,char_seg_labels,seg_labels):
	## setup criterion
	ce_loss = nn.CrossEntropyLoss()
	ctc_loss = nn.CTCLoss(zero_infinity =True)

	# loss on similarity map
	max_len = max(img_lengths[0::2]).item()	
	logits = compute_logits(sims,char_seg_labels, args.nClasses)
	seg_labels = seg_labels[:,:max_len].reshape(-1).long().cuda() + 1
	loss1 = ce_loss(logits, seg_labels)

	# ctc loss on sequence prediction
	labels = labels.view(preds.shape[0],-1)[:,:preds.shape[1]]
	preds_ctc,labels_ctc,input_lens,label_lens = \
		convert_ctc_format(preds,labels,img_lengths[::2],args.alphabet)
	loss2 = ctc_loss(preds_ctc,labels_ctc,input_lens,label_lens)

	return loss1, loss2


if __name__ == '__main__':
	main()
