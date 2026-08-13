import os

import torch
from monai.networks.nets import DynUNet

from .consts import DEEP_SUPR_NUM, PATCH_SIZE, PROPERTIES, SPACING


def get_kernels_strides(sizes=PATCH_SIZE, spacings=SPACING):
    """
    Compute DynUNet kernel sizes and strides for the MindGlide patch size and
    spacing (adapted from the MONAI DynUNet tutorial).
    The patch size in each spatial dimension must be divisible by the product of
    all strides in that dimension, and at least one dimension must be twice the
    product of all strides; otherwise a ValueError is raised.
    """
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size "
                    f"{input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(device, properties=PROPERTIES, checkpoint_path=None, deep_supr_num=DEEP_SUPR_NUM):
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])
    kernels, strides = get_kernels_strides()

    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num,
    )
    net = net.to(device)

    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    return net
