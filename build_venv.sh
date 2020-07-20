export CONDA_ENV_NAME=FontAdaptor
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

pip install numpy==1.16.4 torch==1.2.0 torchvision==0.4.0
pip install -r requirements.txt