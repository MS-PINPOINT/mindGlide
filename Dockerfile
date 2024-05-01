FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

# Temporarily disable NVIDIA repositories
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Update and install wget and gnupg2 without NVIDIA repositories
RUN apt-get update && apt-get install -y gnupg2 wget

# Add the NVIDIA GPG key
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -

# Re-enable NVIDIA repositories
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
 && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# Now it's safe to perform update and install other packages
RUN apt-get update

ARG USER_ID
ARG GROUP_ID
ARG UNAME
RUN groupadd -g $GROUP_ID -o user
RUN useradd -m -u $USER_ID -g $GROUP_ID -o -s /bin/bash $UNAME
USER root
RUN apt-get install -y git
WORKDIR /opt
#ENV PATH="/opt/miniconda3/bin:${PATH}"
#ARG PATH="/opt/miniconda3/bin:${PATH}"
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh \ 
#    &&  bash  Miniconda3-py39_4.11.0-Linux-x86_64.sh -p /opt/miniconda3 -b  \
#    &&  rm -vf Miniconda3-py39_4.11.0-Linux-x86_64.sh
#RUN conda install -y pytorch torchvision -c pytorch 
#RUN python -c 'import torch;print(torch.backends.cudnn.version())'
#RUN apt-get install git
RUN git clone https://github.com/Project-MONAI/MONAI.git  /opt/monai
WORKDIR /opt/monai
RUN git checkout b7403ee5a91caef158181a9dcc2b9b1453f9dbdd
RUN python setup.py install
#RUN conda install -y jupyter matplotlib 
#RUN conda install -y -c conda-forge nibabel 
#RUN conda install -y ignite -c pytorch 
#RUN conda install -y scikit-image
#RUN conda install -y -c conda-forge tensorboard gdown python-lmdb 
RUN echo "export PATH=/opt/miniconda3/bin:${PATH}" >> /etc/profile
USER $UNAME
RUN pip install --upgrade pip
RUN pip install -q "monai-weekly[nibabel, tqdm]"
RUN pip install -q nilearn 
RUN pip install notebook jupyterlab matplotlib
WORKDIR /mnt
#mongo
USER root
#RUN conda install -c anaconda pymongo qgrid dnspython
RUN pip install pymongo  dnspython qgrid scikit-image
RUN pip install pytorch-ignite
##RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
##RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
##RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
##RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
##RUN apt-get update
##
##RUN apt-get install libnccl2 libnccl-dev
RUN pip install ipdb
RUN pip install scipy==1.10.1
RUN git clone https://github.com/Project-MONAI/tutorials.git /opt/monai-tutorials
WORKDIR /opt/monai-tutorials 
RUN ls
RUN git checkout c501cbef2c291b4920b9a8ad3e4a67f334f79f30
COPY  mindGlide/config/task_params.py /opt/monai-tutorials/modules/dynunet_pipeline/
WORKDIR /mnt
RUN pip install nibabel
RUN pip install ipdb 
USER $UNAME
COPY scripts/entrypoint.sh /entrypoint.sh 
COPY ./ /opt/mindGlide
ENTRYPOINT [ "/entrypoint.sh" ]