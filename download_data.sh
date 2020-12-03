mkdir -p data
cd data
wget http://www.robots.ox.ac.uk/~vgg/research/FontAdaptor20/FontSynth_v1.1.tar
tar -xvf FontSynth.tar
rm FontSynth.tar
mkdir -p omniglot-py
cd omniglot-py
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
unzip images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
unzip images_evaluation.zip
rm images_background.zip
rm images_evaluation.zip
