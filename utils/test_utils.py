import os.path as osp

def select_test_font(evalset,root):

	testfonts = []
	if 'Google1k' in evalset:
		fonts = cp.load(open(osp.join(root,'/google1000/google1k_proc/testfonts.pkl'),'rb'))
		if '_' in evalset:
			name = evalset.split('_')[-1]
			testfonts = fonts[name]

		else:
			lines = open(osp.join(root,'/google1000/google1k_proc/testing.txt'),'r').readlines()
			for line in lines:
				volume = line.strip().split('\t')[0]
				if volume not in testfonts:
					testfonts.append(volume)


	elif evalset == 'omniglot':
		testfonts  = np.arange(0,20)

	elif evalset == 'FontSync':
		lines = open( osp.join(root,'gt/test_FontSync.txt'),'r').readlines()
		for line in lines:
			testfonts.append(line.split('/')[-1].replace('.ttf','').replace('\n',''))

	return testfonts

def select_alphabet(lang):

	if lang == 'IT':
		alphabet_gt = '/ derlaiuvtùpfozkchnmsqgòàbèìéjáyxwóíú-'
	elif lang == 'FR':
		alphabet_gt = '/ parvenujqàosdtifcmlégûbhxêùôyèüzwkîâëïò-'
	elif lang =='ES':
		alphabet_gt = '/ bildaquencoshtrzfpgémvyjáxíúóñàеüâèûêùïk-'
	elif lang == 'EN':
		alphabet_gt= '/ abcdefghijklmnopqrstuvwxyz-'
	return alphabet_gt