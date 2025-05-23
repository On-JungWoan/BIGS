FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6"

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.6/

RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY . /home/appuser/Grounded-Segment-Anything/

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano=2.* \
    vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/Grounded-Segment-Anything
RUN pip install -e ./submodules/Grounded-Segment-Anything/segment_anything
# RUN python -m pip install --no-cache-dir -e segment_anything

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    pip install -e ./GroundingDINO/
# RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

WORKDIR /home/appuser
RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

RUN git clone https://github.com/xinyu1205/recognize-anything.git && \
    pip install -r ./recognize-anything/requirements.txt && \
    pip install -e ./recognize-anything/

RUN pip install -U numpy==1.23.1