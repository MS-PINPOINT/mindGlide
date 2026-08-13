"""Unit tests for the volumes tool (fast, no model needed)."""
import nibabel as nib
import numpy as np
import pytest

from mindglide.volumes import calculate_volumes, looks_like_segmentation, volumes_dataframe


def save_seg(path, data, zooms=None):
    img = nib.Nifti1Image(data, np.eye(4))
    if zooms is not None:
        img.header.set_zooms(zooms)
    nib.save(img, path)
    return path


def test_volumes_use_spatial_zooms_only(tmp_path):
    """A 4D image's temporal zoom (TR) must not scale the volumes."""
    data = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    data[:3, :3, :3, 0] = 18  # 27 voxels of label 18
    seg = save_seg(tmp_path / "seg4d.nii.gz", data, zooms=(1, 1, 1, 2.5))
    vols = calculate_volumes(seg)
    assert vols[18] == pytest.approx(27.0)

    # pixdim[4] == 0 (common in real exports) must not zero every volume
    seg0 = save_seg(tmp_path / "seg4d_tr0.nii.gz", data, zooms=(1, 1, 1, 0))
    assert calculate_volumes(seg0)[18] == pytest.approx(27.0)


def test_volumes_anisotropic_voxels(tmp_path):
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[0, 0, 0] = 3
    seg = save_seg(tmp_path / "seg.nii.gz", data, zooms=(1.0, 1.0, 5.0))
    assert calculate_volumes(seg)[3] == pytest.approx(5.0)


def test_volumes_ignore_non_finite(tmp_path, capsys):
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = 2
    seg = save_seg(tmp_path / "segnan.nii.gz", data)
    vols = calculate_volumes(seg)
    assert vols[2] == pytest.approx(1.0)
    assert all(np.isfinite(list(vols.keys())))
    assert "non-finite" in capsys.readouterr().out


def test_dataframe_zero_fills_absent_labels(tmp_path):
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0] = 13
    seg = save_seg(tmp_path / "seg.nii.gz", data)
    df = volumes_dataframe(seg)
    assert len(df) == 20
    assert df.loc[df.Label_ID == 13, "Volume_mm3"].item() == pytest.approx(16.0)
    assert df.loc[df.Label_ID == 5, "Volume_mm3"].item() == 0.0
    assert not df.Volume_mm3.isna().any()


def test_looks_like_segmentation(tmp_path):
    labels = np.zeros((6, 6, 6), dtype=np.uint8)
    labels[:2] = 13
    assert looks_like_segmentation(save_seg(tmp_path / "seg.nii.gz", labels))

    intensity = np.random.default_rng(0).normal(500, 100, (6, 6, 6)).astype(np.float32)
    assert not looks_like_segmentation(save_seg(tmp_path / "raw.nii.gz", intensity))
