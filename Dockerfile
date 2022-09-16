# Base image
# https://hub.docker.com/r/pytorch/pytorch/tags?page=1
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel


# Install 
# Install system packages
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    vim \
    libgl1-mesa-glx \
    screen \
    libssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Install newest CMake

# Install python dependencies
RUN pip3 install \
    Shapely==1.8.0 \
    diskcache==5.2.1 \
    moviepy==1.0.3 \
    pytorch_lightning==1.6.0 \
    tqdm==4.64.0 \
    open3d==0.15.2 \
    Click==7.0 \
    setuptools==59.5.0 \
    PyYAML==6.0

# RUN pip3 install \
#     torch==1.10.1+cu102 \
#     torchvision==0.11.2+cu102 \
#     torchaudio==0.10.1+cu102 \
#     torchtext==0.11.1 \
#     -f https://download.pytorch.org/whl/cu102/torch_stable.html

# setup environment
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Provide a data directory to share data across docker and the host system
RUN mkdir -p /data
RUN mkdir -p /user/dev/dcpcr

WORKDIR /user/dev/dcpcr/
ADD ./setup.py /user/dev/
RUN pip3 install -U -e /user/dev/
USER user



