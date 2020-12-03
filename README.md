# [Adaptive Text Recognition through Visual Matching](http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/)

📋 This repository contains the data and implementation of ECCV2020 paper [Adaptive Text Recognition through Visual Matching](http://www.robots.ox.ac.uk/~vgg/publications/2020/Zhang20/zhang20.pdf)

# Abstract

<p float="center">
  <img src="http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/figs/teaser-hor.png" />
</p>

This work addresses the problems of generalization and flexibility for text recognition in documents.   
We introduce a new model that exploits the repetitive nature of characters in languages, and decouples the visual decoding and linguistic modelling stages through intermediate representations in the form of similarity maps. By doing this, we turn text recognition into a visual matching problem, thereby achieving one-shot sequence recognition.  
It can handle challenges that traditional architectures are unable to solve without expensive retraining, including: (i) it can change the number of classes simply by changing the exemplars; and (ii) it can generalize to novel languages and characters (not in the training data) simply by providing a new glyph exemplar set. We also demonstrate that the model can generalize to unseen fonts without requiring new exemplars from them.


# Getting started
1. Clone this repository
```
git clone https://github.com/Chuhanxx/FontAdaptor.git
```
2. Create conda virtual env and install the requirements  
(This implementation requires CUDA and python > 3.7)
```
cd FontAdaptor
source build_venv.sh
```
3. Download data for training and evalutaion.  
(The dataset contains FontSynth + Omniglot)
```
source download_data.sh
```
4. Download our [pre-trained model](http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/att4_omni.pth) on four font atrributes + Omniglot 

# Test trained models

Test the model using test fonts as exemplars: 

```
python test.py --evalset FontSynth --root ./data --model_folder /PATH/TO/CHECKPOINT 
```

Test the model using randomly chosen training fonts as exemplars 
```
python test.py --evalset FontSynth --root ./data --model_folder /PATH/TO/CHECKPOINT --cross
```

Test the model on Omniglot:

```
python test.py --evalset Omniglot --root ./data --model_folder /PATH/TO/CHECKPOINT 
```

You can visualize the prediction from the model by enabling `--visualize`


# Training 

[** Note our FontSynth dataset has been updated in 04/12/2020 , please download/update it from here](http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/FontSynth_v1.1.tar).

Train the model with English data:
Choose number of attributes by setting `trainset` to `attribute1`,`attribute2`,`attribute3` or `attribute4`

```
python train.py  --trainset attribute4 --data eng --char_aug --name EXP_NAME --root ./data  
```

Train the model with English data + Omniglot:

```
python train.py  --trainset attribute4 --data omnieng --char_aug --name EXP_NAME --root ./data  
```

# Data

Our FontSynth dataset (16GB) can be downloaded directly from [here](http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/FontSynth_v1.1.tar). (updated 03/12/20)

We take 1444 fonts from the [MJSynth dataset](https://www.robots.ox.ac.uk/~vgg/data/text/) and split them into five categories by their appearance attributes as determined from their names: (1) regular, (2) bold, (3) italic, (4) light, and (5) others (i.e., all fonts with none of the first four attributes in their name)  
For train- ing, we select 50 fonts at random from each split and generate 1000 text-line and glyph images for each font. For testing, we use all the 251 fonts in category (5).

<p float="center">
  <img src="http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/figs/font-split.png" width="50%"/>
</p>

The structure of this dataset is:
```
ims/
  font1/
  font2/
  ...
gt/
  train/
    train_regular_50_resample.txt
  test/
  val/
  test_FontSynth.txt
  train_att4.txt
  ...
fontlib/
  googlefontdirectory/
  ...
```
In folder `gt`, there are txt files with lines in the following format:  
```
font_name   img_name   gt_sentence   (H,W)
```
For training, it corresponds to an text-line image with path: `ims/font_name/lines_byclusters/img_name`  
For testing, it corresponds to an text-line image with path: `ims/font_name/test_new/img_name`

`gt/train_att4.txt` and `gt/train_att4.txt` list the fonts selected for training and testing, source files of these fonts can be found in `fontlib`.

# Citation

If you use this code etc., please cite the following paper:

```
@inproceedings{zhang2020Adaptive,
  title={Adaptive Text Recognition through Visual Matching},
  author={Chuhan Zhang and Ankush Gupta and Andrew Zisserman},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

If you have any question, please contact czhang@robots.ox.ac.uk .
