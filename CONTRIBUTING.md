# Contributing to MindGlide

Thanks for your interest! Bug reports, questions, and pull requests are all
welcome.

## Questions and bug reports

- **Questions / usage help**: open a
  [Discussion](https://github.com/MS-PINPOINT/mindGlide/discussions).
- **Bugs**: open an [Issue](https://github.com/MS-PINPOINT/mindGlide/issues)
  with the exact command you ran, the full output, and
  `mindglide --version`. If the problem involves a specific scan, its header
  (`python -c "import nibabel; print(nibabel.load('scan.nii.gz').header)"`)
  helps a lot — never upload patient data.

## Development setup

```bash
git clone https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
pip install -e ".[test]"

pytest                          # fast unit tests (seconds, no model download)
MINDGLIDE_RUN_SLOW=1 pytest -v  # + end-to-end on a public MNI scan
pipx run ruff check inference tests   # lint (CI enforces this)
```

## Pull requests

- Branch from `main`; CI (lint, tests on Python 3.9–3.13, package build) must
  pass.
- **Segmentation output is a contract.** Any change that could touch the
  numerical path must demonstrate byte-identical outputs on the public MNI
  template (see the end-to-end tests) — or clearly declare and justify the
  behaviour change in the PR description.
- Add or update tests for what you change; keep error messages actionable.

## Releases (maintainers)

Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, merge, then create
a GitHub release with tag `v<version>`. CI publishes to PyPI and pushes the
container image to ghcr.io automatically.
