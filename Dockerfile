# Official PyTorch image with CUDA runtime (also works on CPU-only hosts)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/
RUN pip install --no-cache-dir .

# Bake the model weights into the image so the container works offline
RUN mkdir -p /workspace/models && \
    wget -q -O /workspace/models/_20240404_conjurer_trained_dice_7733.pt \
        https://huggingface.co/MS-PINPOINT/mindglide/resolve/main/_20240404_conjurer_trained_dice_7733.pt

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH="/workspace/models/_20240404_conjurer_trained_dice_7733.pt"

ENTRYPOINT ["mindglide"]
CMD ["--help"]
