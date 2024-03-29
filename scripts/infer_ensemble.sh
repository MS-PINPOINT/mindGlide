#run this from host when testing with docker
working_dir="$PWD"
docker run --gpus all --ipc=host --ulimit memlock=-1 \
--shm-size="10gb" --ulimit stack=67108864 -v ${working_dir}:/mnt \
-v /mounts/auto/arman7:/mounts/auto/arman7:shared  \
-v /mounts/auto/arman6:/mounts/auto/arman6:shared  \
--network=host --rm -it -e NCCL_DEBUG=INFO \
--entrypoint /mounts/auto/arman7/workflows/mindGlide/scripts/infer_ensemble_from_container.sh \
armaneshaghi/mind-glide:latest 


