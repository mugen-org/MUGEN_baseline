# Baseline for  MUGEN

[Project Page](https://mugen-org.github.io/)

## Install
Please run the following command to setup the environment.
```
conda create --name mugen python=3.9.5
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.3.3 einops ftfy regex transformers==4.11.3
pip install av==8.0.3
pip install fire soundfile librosa numba unidecode tqdm mpi4py tensorboardX
pip install pycocoevalcap # need to fix the bug mentioned here https://github.com/tylin/coco-caption/pull/35/files
```

## Dataset
Please run the following command to download the dataset.
```
mkdir -p datasets/coinrun
cd datasets/coinrun
wget http://dl.noahmt.com/creativity/data/MUGEN_release/coinrun.zip
unzip coinrun.zip
cd ...
```
For more information, please refer [here](https://mugen-org.github.io/download).

## Model

Run the following command to download the pre-trained checkpoints.
```
mkdir checkpoints
cd checkpoints
wget https://dl.noahmt.com/creativity/data/MUGEN_release/checkpoints.zip
unzip checkpoints
cd ..
```
Please refer [here](retrieval/README.md) for video-audio-text retrieval details and [here](generation/README.md) for video-audio-text generation details.

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@article{hayes2022mugen,
  title={MUGEN: A Playground for Video-Audio-Text Multimodal Understanding and GENeration},
  author={Hayes, Thomas and Zhang, Songyang and Yin, Xi and Pang, Guan and Sheng, Sasha and Yang, Harry and Ge, Songwei and Hu, Qiyuan and Parikh, Devi},
  journal={arXiv preprint arXiv:2204.08058},
  year={2022}
}
```
