"""Fast tests for the public Python API surface (no model download)."""
import nibabel as nib
import numpy as np
import pytest

import mindglide
from mindglide import UsageError, segment


def make_nifti(path):
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)


def test_public_surface():
    assert callable(mindglide.segment)
    assert callable(mindglide.volumes_dataframe)  # lazy attribute
    assert callable(mindglide.calculate_volumes)
    assert issubclass(mindglide.UsageError, Exception)
    with pytest.raises(AttributeError):
        _ = mindglide.does_not_exist


def test_directory_input_requires_output():
    with pytest.raises(UsageError) as e:
        segment(".")
    assert "output_path is required" in str(e.value)


def test_missing_input_raises_usage_error(tmp_path):
    with pytest.raises(UsageError) as e:
        segment(tmp_path / "nope.nii.gz")
    assert "not found" in str(e.value)


def test_refuses_overwriting_input(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    make_nifti(scan)
    with pytest.raises(UsageError) as e:
        segment(scan, scan)
    assert "refusing to overwrite" in str(e.value)


def test_resume_returns_existing_output_without_model(tmp_path):
    """With resume=True and the output already present, segment() returns
    immediately — no model download, no torch import needed."""
    scan = tmp_path / "scan.nii.gz"
    seg = tmp_path / "scan_seg.nii.gz"
    make_nifti(scan)
    make_nifti(seg)
    result = segment(scan, resume=True)  # default output = scan_seg.nii.gz
    assert str(result) == str(seg)


def test_labels_flag_prints_table(capsys):
    import sys as _sys

    from mindglide.infer import main

    argv = _sys.argv
    _sys.argv = ["mindglide", "--labels"]
    try:
        main()
    finally:
        _sys.argv = argv
    out = capsys.readouterr().out
    assert "Lesion" in out and "White_matter" in out and "19" in out
