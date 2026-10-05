"""Exercise working-memory guards through real synthetic NIfTI inputs."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np


def load_pipeline():
    path = Path(__file__).resolve().parents[1] / 'public' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location('browser_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_scan(shape, spacing):
    values = np.linspace(1, 2, shape[2], dtype=np.float32)
    data = np.broadcast_to(values, shape).copy()
    image = nib.Nifti1Image(data, np.diag([*spacing, 1.]))
    image.header.set_xyzt_units('mm')
    return image.to_bytes()


class WorkingMemoryBounds(unittest.TestCase):
    def setUp(self):
        self.pipeline = load_pipeline()

    def tearDown(self):
        self.pipeline.clear()

    def test_padding_expansion_rejected_before_resize(self):
        raw = synthetic_scan((780, 10, 1024), (1., .1, 20.))
        with patch.object(self.pipeline, '_resize', side_effect=AssertionError('resize reached')):
            with self.assertRaisesRegex(ValueError, 'padded working volume'):
                self.pipeline.preprocess(raw)

    def test_anisotropic_intermediate_rejected_before_resize(self):
        raw = synthetic_scan((20, 128, 1024), (20., 1., .1))
        with patch.object(self.pipeline, '_resize', side_effect=AssertionError('resize reached')):
            with self.assertRaisesRegex(ValueError, 'Resampling this scan'):
                self.pipeline.preprocess(raw)

    def test_normal_scan_still_prepares(self):
        raw = synthetic_scan((24, 28, 20), (1., 1., 1.))
        info = json.loads(self.pipeline.preprocess(raw))
        self.assertEqual(info['workingShape'], [24, 28, 20])
        self.assertEqual(info['paddedShape'], [128, 128, 64])
        self.assertEqual(self.pipeline.prepared_image.dtype, np.float32)
        self.assertTrue(np.isfinite(self.pipeline.prepared_image).all())


if __name__ == '__main__':
    unittest.main()
