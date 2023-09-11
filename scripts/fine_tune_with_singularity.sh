#working_dir="$PWD"
#docker run --gpus all --ipc=host --ulimit memlock=-1 \
#--shm-size="10gb" --ulimit stack=67108864 -v ${working_dir}:/mnt \
#-v /mounts/auto/arman7:/mounts/auto/arman7:shared  \
#--network=host --rm -it -e NCCL_DEBUG=INFO \
#--entrypoint /mounts/auto/arman7/workflows/mindGlide/scripts/fine_tune_from_container.sh \
#container_name 
#mind-glide_sep2023.sif
#armaneshaghi/mind-glide:latest 

apptainer run --nv \
--bind ${working_dir}:/mnt,/mounts/auto/arman7:/mounts/auto/arman7 \
--workdir /mnt \
mind-glide_sep2023.sif \
/mounts/auto/arman7/workflows/mindGlide/scripts/fine_tune_from_container.sh
