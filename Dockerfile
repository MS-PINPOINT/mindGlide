# Use the official PyTorch 2.9.1 image with CUDA 12.8 (cu128)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install any system dependencies your project needs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# (Optional) If you need a specific Python version or virtualenv:
# For example, ensure python3.11 is available:
# RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip packages
# Make sure the torch version corresponds to your base image – 
# here it’s already installed via the base image, but you might reinstall/upgrade other libs.
COPY . /workspace/
RUN pip install --no-cache-dir .
    
RUN mkdir -p /workspace/models && \
    wget -O /workspace/models/_20240404_conjurer_trained_dice_7733.pt \
        https://huggingface.co/MS-PINPOINT/mindglide/resolve/main/_20240404_conjurer_trained_dice_7733.pt
ENV PYTHONUNBUFFERED=1 \
    # any other environment variables your app needs \
    CUDA_LAUNCH_BLOCKING=1 \
    MODEL_PATH="/workspace/models/_20240404_conjurer_trained_dice_7733.pt"

