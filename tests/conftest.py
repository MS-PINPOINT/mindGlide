import os
import urllib.request

import numpy as np
import nibabel as nib
import pytest

# Public brain MRI used for end-to-end tests: the MNI152 (2009c asymmetric)
# T1 template from TemplateFlow (CC0-licensed, no registration needed).
MNI_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
    "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
)


@pytest.fixture(scope="session")
def mni_t1(tmp_path_factory):
    """Download (once per session) a public MNI152 T1 template, 2 mm resolution."""
    cache = tmp_path_factory.mktemp("data") / "mni_t1_2mm.nii.gz"
    urllib.request.urlretrieve(MNI_URL, cache)
    return cache


@pytest.fixture()
def synthetic_brain(tmp_path):
    """A small synthetic 'brain': a bright ellipsoid on dark background."""
    shape = (48, 56, 40)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij",
    )
    data = ((xx ** 2 + yy ** 2 + zz ** 2) < 0.6).astype(np.float32) * 100
    data += np.random.default_rng(0).normal(0, 1, shape).astype(np.float32)
    path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MINDGLIDE_RUN_SLOW"):
        return
    skip_slow = pytest.mark.skip(
        reason="slow end-to-end test; set MINDGLIDE_RUN_SLOW=1 to run"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
