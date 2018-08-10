import os
from setuptools import setup, find_packages
import torch

curdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      'apex_utils')
os.chdir(curdir)

if not torch.cuda.is_available():
    print("Warning: Torch did not find available GPUs on this system.\n",
          "If your intention is to cross-compile, this is not an error.")

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("APEx requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

print("Building module.")
setup(
    name='apex', version='0.1',
#    ext_modules=[cuda_ext,],
    description='PyTorch Extensions written by NVIDIA',
    packages=find_packages(where='.',
                           exclude=(
                               "build",
                               "csrc",
                               "include",
                               "tests",
                               "dist",
                               "docs",
                               "tests",
                               "examples",
                               "apex.egg-info",
                           )),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "unidecode",
        "seaborn",
    ]
)
