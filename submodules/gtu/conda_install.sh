#!/bin/bash

# CUDA_VERSION : 11.8

conda create -n gtu python==3.9
conda activate gtu

# torch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install chamfer-distance
pip install git+'https://github.com/otaheri/chamfer_distance'

# Install diff-gaussian-rasterizer (Modified to render alpha-channel (also back-propabable))
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Install pytorch3D (It's just for debug visualization.)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

# Install xformers
pip install -U xformers==0.0.22

# Install diffusers
pip install "diffusers[torch]"

# Install other dependencies
pip install -r requirements_basic.txt

# fix
pip install diffusers==0.29.0








pip install xformers --index-url https://download.pytorch.org/whl/cu118

# Install diffusers
pip install diffusers==0.28.2
pip install huggingface_hub==0.24.7

# Install other dependencies
pip install -r requirements_basic.txt

#
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html

# 
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Install chamfer-distance
pip install git+'https://github.com/otaheri/chamfer_distance'

