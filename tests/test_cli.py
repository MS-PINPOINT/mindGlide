"""Unit tests for the command-line interface plumbing (fast, no model needed)."""
import os

import nibabel as nib
import numpy as np
import pytest

from mindglide.infer import collect_io, is_nifti, nifti_stem, parse_args, resolve_model_path


def make_nifti(path):
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)


class TestHelpers:
    def test_is_nifti(self):
        assert is_nifti("scan.nii")
        assert is_nifti("scan.nii.gz")
        assert is_nifti("SCAN.NII.GZ")
        assert not is_nifti("scan.txt")
        assert not is_nifti("scan")

    def test_nifti_stem(self):
        assert nifti_stem("sub-01_T1w.nii.gz") == ("sub-01_T1w", "nii.gz")
        assert nifti_stem("scan.nii") == ("scan", "nii")
        assert nifti_stem("a.b.c.nii.gz") == ("a.b.c", "nii.gz")


class TestParseArgs:
    def test_basic(self):
        args = parse_args(["-i", "a.nii.gz", "-o", "b.nii.gz"])
        assert args.i == "a.nii.gz" and args.o == "b.nii.gz"
        assert args.device == "auto"

    def test_flag_spelling_variants(self):
        # both hyphen and underscore spellings are accepted
        args = parse_args(["-i", "a", "-o", "b", "--sw-batch-size", "2",
                           "--no-klc", "--no_reorient", "--model-path", "m.pt"])
        assert args.sw_batch_size == 2
        assert args.no_klc and args.no_reorient
        assert args.model_path == "m.pt"


class TestCollectIO:
    def test_single_file(self, tmp_path):
        inp = tmp_path / "scan.nii.gz"
        make_nifti(inp)
        out = tmp_path / "new_dir" / "seg.nii.gz"
        ins, outs = collect_io(str(inp), str(out))
        assert ins == [str(inp)] and outs == [str(out)]
        # output parent directory is created up front
        assert out.parent.is_dir()

    def test_single_file_output_to_existing_dir(self, tmp_path):
        inp = tmp_path / "scan.nii.gz"
        make_nifti(inp)
        ins, outs = collect_io(str(inp), str(tmp_path))
        assert outs == [str(tmp_path / "scan_seg.nii.gz")]

    def test_missing_input_file_errors_cleanly(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            collect_io(str(tmp_path / "nope.nii.gz"), str(tmp_path / "o.nii.gz"))
        assert "not found" in str(e.value)

    def test_non_nifti_output_errors_cleanly(self, tmp_path):
        inp = tmp_path / "scan.nii.gz"
        make_nifti(inp)
        with pytest.raises(SystemExit) as e:
            collect_io(str(inp), str(tmp_path / "seg.txt"))
        assert ".nii" in str(e.value)

    def test_directory_mode_skips_clutter(self, tmp_path, capsys):
        inp = tmp_path / "in"
        inp.mkdir()
        make_nifti(inp / "scan1.nii.gz")
        make_nifti(inp / "scan2.nii")
        # clutter that used to crash the old parser (no dot in the name, subdir, txt)
        (inp / "README").write_text("hello")
        (inp / "notes").mkdir()
        (inp / "scan3.txt").write_text("not a scan")

        out = tmp_path / "out"
        ins, outs = collect_io(str(inp), str(out))
        assert sorted(os.path.basename(f) for f in ins) == ["scan1.nii.gz", "scan2.nii"]
        assert sorted(os.path.basename(f) for f in outs) == ["scan1_seg.nii.gz", "scan2_seg.nii"]
        assert out.is_dir()

    def test_directory_without_niftis_errors_cleanly(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        (inp / "README").write_text("hello")
        with pytest.raises(SystemExit) as e:
            collect_io(str(inp), str(tmp_path / "out"))
        assert "no NIfTI" in str(e.value)

    def test_directory_input_with_file_output_errors_cleanly(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_nifti(inp / "scan1.nii.gz")
        out = tmp_path / "existing.nii.gz"
        make_nifti(out)
        with pytest.raises(SystemExit) as e:
            collect_io(str(inp), str(out))
        assert "directory" in str(e.value)

    def test_resume_skips_existing(self, tmp_path):
        inp = tmp_path / "in"
        out = tmp_path / "out"
        inp.mkdir()
        out.mkdir()
        make_nifti(inp / "scan1.nii.gz")
        make_nifti(inp / "scan2.nii.gz")
        make_nifti(out / "scan1_seg.nii.gz")
        ins, _ = collect_io(str(inp), str(out), resume=True)
        assert [os.path.basename(f) for f in ins] == ["scan2.nii.gz"]


class TestResolveModelPath:
    def test_cli_missing_file_errors_cleanly(self):
        with pytest.raises(SystemExit) as e:
            resolve_model_path("/does/not/exist.pt")
        assert "not found" in str(e.value)

    def test_cli_beats_env(self, tmp_path, monkeypatch):
        cli_ckpt = tmp_path / "cli.pt"
        env_ckpt = tmp_path / "env.pt"
        cli_ckpt.write_bytes(b"x")
        env_ckpt.write_bytes(b"x")
        monkeypatch.setenv("MODEL_PATH", str(env_ckpt))
        assert resolve_model_path(str(cli_ckpt)) == cli_ckpt

    def test_env_used_when_no_cli(self, tmp_path, monkeypatch):
        env_ckpt = tmp_path / "env.pt"
        env_ckpt.write_bytes(b"x")
        monkeypatch.setenv("MODEL_PATH", str(env_ckpt))
        assert resolve_model_path(None) == env_ckpt


class TestOverwriteGuards:
    def test_single_file_same_input_output_refused(self, tmp_path):
        scan = tmp_path / "scan.nii.gz"
        make_nifti(scan)
        with pytest.raises(SystemExit) as e:
            collect_io(str(scan), str(scan))
        assert "refusing to overwrite" in str(e.value)

    def test_seg_inputs_skipped_in_directory_mode(self, tmp_path, capsys):
        inp = tmp_path / "d"
        inp.mkdir()
        make_nifti(inp / "scan.nii.gz")
        make_nifti(inp / "scan_seg.nii.gz")
        ins, outs = collect_io(str(inp), str(inp))
        assert [os.path.basename(f) for f in ins] == ["scan.nii.gz"]
        assert "previous segmentations" in capsys.readouterr().out

    def test_case_variant_inputs_colliding_on_output_refused(self, tmp_path):
        inp = tmp_path / "d"
        inp.mkdir()
        make_nifti(inp / "scan.nii.gz")
        make_nifti(inp / "scan.NII.GZ")
        with pytest.raises(SystemExit) as e:
            collect_io(str(inp), str(tmp_path / "out"))
        assert "same output file" in str(e.value)


class TestResumeSemantics:
    def test_resume_is_extension_precise(self, tmp_path):
        inp = tmp_path / "in"
        out = tmp_path / "out"
        inp.mkdir()
        out.mkdir()
        make_nifti(inp / "scan.nii")
        make_nifti(inp / "scan.nii.gz")
        make_nifti(out / "scan_seg.nii")  # only the .nii variant is done
        ins, _ = collect_io(str(inp), str(out), resume=True)
        assert [os.path.basename(f) for f in ins] == ["scan.nii.gz"]

    def test_resume_ignores_unrelated_seg_named_files(self, tmp_path):
        inp = tmp_path / "in"
        out = tmp_path / "out"
        inp.mkdir()
        out.mkdir()
        make_nifti(inp / "notes.nii.gz")
        (out / "notes_seg.txt").write_text("a QC log, not a segmentation")
        ins, _ = collect_io(str(inp), str(out), resume=True)
        assert [os.path.basename(f) for f in ins] == ["notes.nii.gz"]

    def test_resume_single_file_skips_existing_output(self, tmp_path, capsys):
        scan = tmp_path / "scan.nii.gz"
        seg = tmp_path / "seg.nii.gz"
        make_nifti(scan)
        make_nifti(seg)
        ins, outs = collect_io(str(scan), str(seg), resume=True)
        assert ins == [] and outs == []
        assert "skipping" in capsys.readouterr().out

    def test_resume_does_not_mask_empty_directory_error(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        (inp / "README").write_text("no niftis here")
        with pytest.raises(SystemExit) as e:
            collect_io(str(inp), str(tmp_path / "out"), resume=True)
        assert "no NIfTI" in str(e.value)


class TestArgValidation:
    def test_directory_input_with_nifti_named_output_refused(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_nifti(inp / "scan.nii.gz")
        with pytest.raises(SystemExit) as e:
            collect_io(str(inp), str(tmp_path / "segs.nii.gz"))
        assert "output directory" in str(e.value)

    def test_sw_batch_size_must_be_positive(self, capsys):
        for bad in ["0", "-2"]:
            with pytest.raises(SystemExit):
                parse_args(["-i", "a.nii.gz", "-o", "b.nii.gz", "--sw-batch-size", bad])

    def test_help_lists_hyphenated_spellings_first(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])
        helptext = capsys.readouterr().out
        assert "--model-path" in helptext and "--sw-batch-size" in helptext
        assert helptext.index("--model-path") < helptext.index("--model_path")
