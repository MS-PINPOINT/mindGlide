#FROM nvcr.io/nvidia/pytorch:23.01-py3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM nvcr.io/nvidia/kaldi:22.12-py3
ENV DEBIAN_FRONTEND=noninteractive
ARG USER_ID
ARG GROUP_ID
ARG UNAME

# Create user and group with specified IDs
RUN groupadd -g $GROUP_ID -o $UNAME && \
    useradd -m -u $USER_ID -g $GROUP_ID -o -s /bin/bash $UNAME

# Install dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update --allow-unauthenticated && \
    apt-get install -y git

# Set up the virtual environment
WORKDIR /opt
RUN mkdir /opt/virtualenv && \
    python3 -m venv /opt/virtualenv
COPY requirements.txt /opt/virtualenv
RUN /opt/virtualenv/bin/pip install --upgrade pip && \
    /opt/virtualenv/bin/pip install -r /opt/virtualenv/requirements.txt

# Clone the MONAI tutorials repository
RUN git clone https://github.com/Project-MONAI/tutorials.git /opt/monai-tutorials
WORKDIR /opt/monai-tutorials
RUN git checkout c501cbef2c291b4920b9a8ad3e4a67f334f79f30

# Copy configuration file
COPY mindGlide/config/task_params.py /opt/monai-tutorials/modules/dynunet_pipeline/

# Copy entrypoint script and make it executable
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER $UNAME

# Set working directory and copy application files
WORKDIR /mnt
COPY ./ /opt/mindGlide

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
