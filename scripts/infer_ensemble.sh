#run this from host
working_dir="$PWD"
docker run --gpus all --ipc=host --ulimit memlock=-1 \
--shm-size="10gb" --ulimit stack=67108864 -v ${working_dir}:/mnt \
-v /mounts/auto/arman7:/mounts/auto/arman7:shared  \
--network=host --rm -it -e NCCL_DEBUG=INFO \
armaneshaghi/monai:latest \
/mounts/auto/arman7/workflows/mindGlide/infer_ensemble_from_container.sh
