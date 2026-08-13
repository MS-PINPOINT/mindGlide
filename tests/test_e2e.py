"""End-to-end tests: run the real CLI on public data (MNI152 template).

These download the model weights (~123 MB, cached by Hugging Face Hub) and run
full inference, so they are skipped unless MINDGLIDE_RUN_SLOW=1 is set:

    MINDGLIDE_RUN_SLOW=1 pytest tests/test_e2e.py -v
"""
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch


def run_cli(*cli_args):
    return subprocess.run(
        [sys.executable, "-m", "mindglide.infer", *cli_args],
        capture_output=True, text=True, timeout=1800,
    )


def check_segmentation(seg_path, src_path, min_labels=10):
    """The segmentation must match the source scan's grid and look like a brain."""
    seg = nib.load(seg_path)
    src = nib.load(src_path)
    assert seg.shape == src.shape, "segmentation shape must match input"
    assert np.allclose(seg.affine, src.affine, atol=1e-4), \
        "segmentation affine must match input"
    labels = np.unique(seg.get_fdata())
    assert labels.max() <= 19, "labels must be within the MindGlide label set"
    assert len(labels) >= min_labels, \
        f"expected a rich segmentation, got only labels {labels}"


@pytest.mark.slow
class TestEndToEnd:
    def test_cpu_single_file(self, mni_t1, tmp_path):
        out = tmp_path / "seg_cpu.nii.gz"
        result = run_cli("-i", str(mni_t1), "-o", str(out), "--device", "cpu")
        assert result.returncode == 0, result.stdout + result.stderr
        check_segmentation(out, mni_t1)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
    def test_gpu_single_file(self, mni_t1, tmp_path):
        out = tmp_path / "seg_gpu.nii.gz"
        result = run_cli("-i", str(mni_t1), "-o", str(out), "--device", "cuda")
        assert result.returncode == 0, result.stdout + result.stderr
        check_segmentation(out, mni_t1)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
    def test_gpu_directory_mode_with_clutter(self, mni_t1, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        (inp / "scan1.nii.gz").write_bytes(mni_t1.read_bytes())
        (inp / "README").write_text("clutter that must be ignored")
        (inp / "subdir").mkdir()

        out = tmp_path / "out"
        result = run_cli("-i", str(inp), "-o", str(out))
        assert result.returncode == 0, result.stdout + result.stderr
        check_segmentation(out / "scan1_seg.nii.gz", mni_t1)

    def test_failed_scan_exits_nonzero(self, tmp_path):
        """A corrupt NIfTI must produce a non-zero exit code, not 'Inference complete'."""
        bad = tmp_path / "bad.nii.gz"
        bad.write_bytes(b"this is not a nifti file")
        out = tmp_path / "seg.nii.gz"
        result = run_cli("-i", str(bad), "-o", str(out), "--device", "cpu")
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "unreadable" in output and "Finished with errors" in output

    def test_truncated_payload_does_not_kill_batch(self, mni_t1, tmp_path):
        """A file with a valid header but corrupt payload passes the header
        preflight; it must fail cleanly inside the loop, and the other scan in
        the batch must still be segmented."""
        inp = tmp_path / "in"
        inp.mkdir()
        (inp / "good.nii.gz").write_bytes(mni_t1.read_bytes())
        whole = mni_t1.read_bytes()
        (inp / "truncated.nii.gz").write_bytes(whole[: len(whole) // 2])

        out = tmp_path / "out"
        result = run_cli("-i", str(inp), "-o", str(out), "--device", "cpu")
        output = result.stdout + result.stderr
        assert result.returncode != 0
        # The run must survive the loader failure: the failure is attributed to
        # the bad file, the summary still prints, and the good scan completes.
        # (torch embeds the worker's traceback in the exception text we relay,
        # so we assert on the summary rather than on the absence of tracebacks.)
        assert "Finished with errors" in result.stdout
        assert "truncated.nii.gz" in result.stdout
        check_segmentation(out / "good_seg.nii.gz", mni_t1)
