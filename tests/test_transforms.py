"""Fast unit coverage of the preprocessing pipeline on a synthetic phantom."""
import numpy as np

from mindglide.transforms import get_transforms


def test_transform_pipeline_on_phantom(synthetic_brain):
    out = get_transforms()({"image": str(synthetic_brain)})

    # Preprocessing contract that infer.main relies on:
    for key in ("image", "bbox", "original_shape", "crop_shape",
                "resample_flag", "anisotropy_flag", "output_affine"):
        assert key in out, f"missing key: {key}"

    # 1 mm isotropic phantom: no resampling, foreground crop within bounds.
    assert bool(out["resample_flag"]) is False
    assert bool(out["anisotropy_flag"]) is False
    (h0, w0, d0), (h1, w1, d1) = out["bbox"]
    assert (h1 - h0, w1 - w0, d1 - d0) == tuple(out["image"].shape[1:])
    assert list(out["original_shape"]) == [48, 56, 40]
    # normalised foreground: roughly zero-mean, unit-ish std
    img = np.asarray(out["image"])
    assert abs(img[img != 0].mean()) < 1.0
