#!/bin/bash

conda activate bigs

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

pip install -r requirements.txt
pip install git+https://github.com/mattloper/chumpy.git

pip install git+'https://github.com/otaheri/chamfer_distance'

# (optional) for sds loss
pip install -U xformers==0.0.22 torch==2.0.1
pip install "diffusers[torch]==0.24.0"