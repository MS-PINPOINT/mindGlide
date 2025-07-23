#FROM nvcr.io/nvidia/pytorch:23.01-py3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM nvcr.io/nvidia/kaldi:22.12-py3
ENV DEBIAN_FRONTEND=noninteractive
ARG USER_ID
ARG GROUP_ID
ARG UNAME
RUN groupadd -g $GROUP_ID -o $UNAME
RUN useradd -m -u $USER_ID -g $GROUP_ID -o -s /bin/bash $UNAME
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update --allow-unauthenticated
RUN apt-get install -y git
WORKDIR /opt
RUN mkdir /opt/virtualenv
RUN python3 -m venv /opt/virtualenv
COPY requirements.txt /opt/virtualenv
RUN /opt/virtualenv/bin/pip install --upgrade pip
RUN /opt/virtualenv/bin/pip install -r /opt/virtualenv/requirements.txt
RUN git clone https://github.com/Project-MONAI/tutorials.git /opt/monai-tutorials
WORKDIR /opt/monai-tutorials 
RUN git checkout c501cbef2c291b4920b9a8ad3e4a67f334f79f30
COPY mindGlide/config/task_params.py /opt/monai-tutorials/modules/dynunet_pipeline/
WORKDIR /mnt
USER $UNAME
COPY scripts/entrypoint.sh /entrypoint.sh 
COPY ./ /opt/mindGlide
ENTRYPOINT [ "/entrypoint.sh" ]
