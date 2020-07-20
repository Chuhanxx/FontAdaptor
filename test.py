import os
import argparse
import scipy.misc
import scipy
import torch
import torchvision

import numpy as np
import _pickle as cp
import os.path as osp

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn.init as init


from dataloader import*
from utils.utils import *
from utils.test_utils import *
from model.model import GlyphAdaptor
ImageFile.LOAD_TRUNCATED_IMAGES = True

def worker_init_fn(worker_id):                                                          
	np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
	parser = argparse.ArgumentParser(description='GlyphAdaptor')

	## data setting 
	parser.add_argument('--root', default='./data',type=str)
	parser.add_argument('--load_height', default=32, type=int)
	parser.add_argument('--evalset', default='FontSynth', type=str,help='FontSynth,Omniglot')

	parser.add_argument("--val", dest="val", action = 'store_true')
	parser.add_argument("--visualize", dest="visualize", action = 'store_true')
	parser.add_argument("--cross", action = 'store_true',help="use training font as reference font")
	parser.add_argument('--lang', default="EN", type=str,help='language for testing in Google1k ')

	# model params
	parser.add_argument("--gpus", dest="gpu", default="0", type=str)
	parser.add_argument('--d_model', default=360, type=int, help='dimension of transformer')
	parser.add_argument('--num_workers', default=4, type=int) 

	## model setting
	parser.add_argument('--alphabet', default='/ abcdefghijklmnopqrstuvwxyz-', type=str)
	parser.add_argument('--pretrained_name', default='att4_omni', type=str)
	parser.add_argument("--TPS", dest="TPS", action = 'store_true')


	## output setting
	parser.add_argument('--model_folder', default='', type=str)

	args = parser.parse_args()


	if osp.exists(args.model_folder) == False:
		raise Exception('directory for pretrained model is not found')

	result_dir = osp.join('./results')
	if osp.exists(result_dir) == False:
		os.mkdir(result_dir)


	args.nClasses = len(args.alphabet)
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.FloatTensor')



	###### setup fonts and alphabet converter ######
	converter = strLabelConverter(args.alphabet)
	if args.evalset =='FontSynth':
		testfonts = select_test_font(args.evalset,args.root)
	elif args.evalset == 'Omniglot':
		testfonts  = np.arange(0,20)
	else:
		raise Exception('test dataset is not recognized')

	args.alphabet_gt=select_alphabet(args.lang)
	converter_gt = strLabelConverter(args.alphabet_gt)
	if args.cross:
		ref_fonts = []
		ref_list = open(osp.join(args.root,'gt','ref_10_random_fonts.txt'),'r').readlines()
		for line in ref_list:
			ref_fonts.append(line.split('/')[-1].replace('.ttf','').replace('\n',''))
	else:
		ref_fonts= ['']



	###### setup model ######
	net = GlyphAdaptor(args)
	net = torch.nn.DataParallel(net).to(device)
	if args.pretrained_name:
		resume_file = osp.join(args.model_folder,args.pretrained_name+'.pth')
		checkpoint = torch.load(resume_file)
		net.load_state_dict(checkpoint['model_state_dict'],strict=True)
	else:
		raise Exception('checkpoint not specified')


	total_acc,total_cer,total_wer = 0,0,0
	###### start testing ######
	print('======== testing starts ========')
	cudnn.benchmark = True
	net.eval()

	for font in ref_fonts:
		args.ref_font = font
		if args.cross:
			print('======== Using %s as reference font========'%args.ref_font)

		total_correct,cers,wers,counter = 0,0,0,0
		for test_font in testfonts:
			args.fontname = test_font


			if args.evalset == 'FontSynth':
				testset =  TestLoader(args,aug= False)
			elif args.evalset == 'Omniglot':
				testset =  Omniglot(args,root=args.root,alpha_ind = test_font ,background=False,size = 500)


			seq_sampler = data.SequentialSampler(testset)
			test_loader = data.DataLoader(testset, 1, num_workers=args.num_workers,
										  sampler = seq_sampler,pin_memory=True, collate_fn=text_collate_eval,drop_last=False,worker_init_fn=worker_init_fn)
			cer_font,acc_font,wer_font,font_logs = [],[],[],[]
			for index, sample in enumerate(test_loader):
				imgs, char_seg_label, labels, length, alphabet= sample

				if args.evalset != 'Omniglot':
					args.alphabet = alphabet

				converter = strLabelConverter(args.alphabet)
				imgs = torch.transpose(imgs.unsqueeze(1),2,3)
				imgs = imgs.float().to(device)
				labels = labels.long().to(device) #[batch*len]

				preds,sims,final_conv = net(imgs,char_seg_label,length.long(),char_seg_label) #[batch,len,classes]
				correct,cer,wer,pred_str,gt_str, _ = lex_free_acc(preds,labels,converter,'ctc',converter_gt = converter_gt)
				probs = torch.softmax(preds[0],dim=-1).detach().cpu().numpy()
				labels = labels.view(preds.shape[0],-1)
				label = labels[0].detach().cpu().numpy()

				counter += 1

				if args.visualize:
					print(pred_str[0],'---',gt_str[0])

				cer_font.append(cer)
				wer_font.append(wer)
				acc_font.append(correct)

				total_correct += correct
				cers += cer 
				wers += wer

			font_summary = str(args.fontname)+'\t CER: '+str(np.mean(cer_font))+'\t WER: '+str(np.mean(wer_font))
			print(font_summary)
			font_logs.append(font_summary)

		total_acc += total_correct*1.0/counter
		total_cer += cers*1.0 / counter
		total_wer += wers*1.0 / counter

		if not args.cross:
			args.ref_font = 'itself' 

		total_summary = 'ref_font=%s cer = %f wer = %f accuracy=%f total_samples= %d'%(args.ref_font,total_cer,total_wer,total_acc,counter)
		print('----------')
		print(total_summary)
		print('----------')


		if args.cross:
			result_file= open(osp.join(result_dir,args.ref_font +'_cross_' + args.evalset+'_acc.txt'),'w')
		else:
			result_file= open(osp.join(result_dir,args.ref_font +'_' + args.evalset+'_acc.txt'),'w')

		for font_summary in font_logs:
			result_file.write(font_summary)
		result_file.write('======================================================')
		result_file.write(total_summary)
		result_file.close() 

	ref_total_cer = total_cer / len(ref_fonts)
	ref_total_wer = total_wer / len(ref_fonts)
	ref_total_acc = total_acc / len(ref_fonts)

	if args.cross:

		print('======================================================')
		print('Total results over all reference fonts')
		print('cer = %f wer = %f accuracy=%f  total_ref_fonts=%d '%(ref_total_cer,ref_total_wer,ref_total_acc,len(ref_fonts)))
		print('======================================================')


if __name__ == '__main__':
	main()
