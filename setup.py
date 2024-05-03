import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='FamDeploy',
    py_modules=["deploy"],
    version='0.0.0',
    packages=find_packages(),
    description='',
    long_description=read('README.md'),
    author='Jinliang Zheng, etc.',
    install_requires=[
        'torch',
        'torchvision',
        'timm',
        'mmengine',
        'tqdm',
        'numpy',
        'transformers>=4.40.0',
        'gdown',
        'opencv-python',
        'flash-attn'
    ]
)