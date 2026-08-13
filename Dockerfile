# Official PyTorch image with CUDA runtime (also works on CPU-only hosts).
# Digest-pinned so rebuilds are reproducible (tag: 2.9.1-cuda12.8-cudnn9-runtime).
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385

# Model checkpoint: pinned to an exact model-repo commit and checksum so an
# image rebuild can never silently bake different weights. Keep these three
# values in sync with HF_MODEL_* in inference/mindglide/infer.py.
ARG MODEL_FILE=_20240404_conjurer_trained_dice_7733.pt
ARG MODEL_REVISION=a1969821c0a4a37ae54f649a9a0c6fd1b8a48e26
ARG MODEL_SHA256=881e30efd9444a25ee70c01d795dd9fb21ac750a48f1ba8070fcd79fb75e76ca

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bake the model weights first: this layer never changes, so source edits do
# not re-download 123 MB.
RUN mkdir -p /workspace/models && \
    wget -q -O "/workspace/models/${MODEL_FILE}" \
        "https://huggingface.co/MS-PINPOINT/mindglide/resolve/${MODEL_REVISION}/${MODEL_FILE}" && \
    echo "${MODEL_SHA256}  /workspace/models/${MODEL_FILE}" | sha256sum -c -

COPY . /workspace/
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH="/workspace/models/${MODEL_FILE}"

# Run as a non-root user so bind-mounted outputs are not root-owned on the
# host. For exact host-uid ownership, prefer: docker run --user $(id -u):$(id -g)
RUN useradd --create-home mindglide
USER mindglide

ENTRYPOINT ["mindglide"]
CMD ["--help"]
