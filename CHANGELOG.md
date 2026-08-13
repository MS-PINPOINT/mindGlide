# Changelog

All notable changes to MindGlide are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- README "any sequence, any quality" hero gallery: real MindGlide
  segmentations of five openly licensed public scans — 3 T T2w, 3D FLAIR,
  7 T PD, a thick-slice clinical 2D FLAIR with lesions, and a T2w from a
  64 mT portable scanner (`assets/any_sequence_any_quality.png`).

### Changed

- Terminology: "any modality" → "any sequence" across the README, package
  metadata, CLI help, and quickstart notebook (T1/T2/FLAIR/PD are MRI
  sequences; modality would imply CT/PET too).

## [1.3.0] - 2026-08-13

### Added

- Python API: `from mindglide import segment` — segment scans from scripts
  and pipelines with the same (byte-identical) results as the CLI; raises
  `mindglide.UsageError` for invalid usage. `volumes_dataframe` /
  `calculate_volumes` are importable from the package root.
- Cohort volumes: `mindglide-volumes <directory>` writes one combined
  long-format CSV (Scan, Label_ID, Region_Name, Volume_mm3) for a whole
  folder of segmentations; also available as
  `mindglide.volumes.cohort_dataframe`.
- `mindglide --labels` prints the label code / region name table.
- Zero-install Colab quickstart notebook (`examples/`), viewer colormaps for
  ITK-SNAP / FSLeyes / freeview (`labels/`), `CITATION.cff` (GitHub "Cite
  this repository"), this changelog, and a contributor guide.
- Prebuilt container images published to GitHub Container Registry
  (`ghcr.io/ms-pinpoint/mindglide`) on every release.

## [1.2.0] "Aegis" - 2026-08-13

Hardening, packaging, and automation release — dedicated to protecting your
data. Segmentation outputs are byte-identical to 1.1.0. First release on PyPI.

### Added

- First PyPI release: `pip install mindglide`.
- Real Python package: `mindglide.__version__` and `python -m mindglide`
  support, SPDX license metadata, complete sdist, declared dependencies.
- GitHub Actions CI: lint + tests on Python 3.9–3.13 and package build on
  every push, plus a weekly end-to-end segmentation of public MNI data.

### Changed

- Model download is pinned to an exact Hugging Face revision and verified by
  checksum (the Docker image verifies weights by SHA-256 and runs as
  non-root).
- `--resume` is now exact: it checks for the precise expected output file
  (extension included), works for single files too, and can no longer be
  fooled by unrelated `*_seg.*` files.

### Fixed

- Data-loss guard: MindGlide refuses to overwrite input scans — `-o`
  resolving onto an input file (including cross-collisions in folder mode)
  errors instead of silently replacing the MRI with the label map.
- Atomic output writes: interrupted runs can no longer leave truncated
  `_seg` files that `--resume` would treat as done.
- Corrupt-file resilience: one corrupt file no longer kills a batch run —
  failures are attributed and reported per file, the rest of the batch
  completes, and the exit code reflects the failures.
- `mindglide-volumes`: correct volumes for 4D label images (spatial zooms
  only), NaN filtering, and a warning when the input doesn't look like a
  segmentation.

## 1.1.0 - 2026-08-12

### Added

- `--device {auto,cpu,cuda,mps}` flag with a CUDA smoke test that falls back
  to CPU when the installed PyTorch build cannot actually run on the GPU.
- Test suite (fast unit tests plus opt-in end-to-end runs).
- README quickstart.

### Changed

- CLI hardening: clear error messages and a non-zero exit code on failure.

### Fixed

- Compatibility with MONAI 1.6.

### Removed

- Legacy container-based training/fine-tuning pipeline, preserved at the
  [`legacy-container`](https://github.com/MS-PINPOINT/mindGlide/tree/legacy-container)
  tag.

## [1.0.1] - 2025-01-23

- Initial public release of the pip package.

## [1.0.0] - 2025-01-23

- Initial public release of the inference CLI.

[1.3.0]: https://github.com/MS-PINPOINT/mindGlide/releases/tag/v1.3.0
[1.2.0]: https://github.com/MS-PINPOINT/mindGlide/releases/tag/v1.2.0
[1.0.1]: https://github.com/MS-PINPOINT/mindGlide/releases/tag/v1.0.1
[1.0.0]: https://github.com/MS-PINPOINT/mindGlide/releases/tag/v1.0.0
