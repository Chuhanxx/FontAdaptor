import os.path as osp

def select_test_font(evalset,root):

	testfonts = []
	if evalset == 'omniglot':
		testfonts  = np.arange(0,20)

	elif evalset == 'FontSynth':
		lines = open( osp.join(root,'gt/test_FontSynth.txt'),'r').readlines()
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