import os
from setuptools import setup, find_packages
import torch

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("Sentiment Discovery requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

print("Building module.")
setup(
    name='sentiment_discovery', version='0.4',
#    ext_modules=[cuda_ext,],
    description='PyTorch Extensions written by NVIDIA',
    packages=find_packages(where='.'),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "unidecode",
        "seaborn",
        "sentencepiece",
        "emoji"
    ]
)
