"""Validate browser preprocessing/recovery and full public-template inference.

Run in an isolated environment with the repository's inference dependencies.
Only public MNI data and generated phantoms are used. CUDA is optional.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import onnxruntime as ort
import torch
from monai.inferers import SlidingWindowInferer
from monai.data.utils import dense_patch_slices, compute_importance_map
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'inference'))
from mindglide.transforms import get_transforms, recovery_prediction, keep_largest_component
from mindglide.network import get_network
from mindglide.consts import PATCH_SIZE
from mindglide.infer import HF_REPO_ID, HF_MODEL_FILENAME, HF_MODEL_REVISION


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location('browser_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_recovery(labels, meta, affine):
    prediction = np.eye(20, dtype=np.uint8)[labels].transpose(3, 0, 1, 2)
    if meta['resample_flag']:
        prediction = recovery_prediction(prediction, [20, *meta['crop_shape']], meta['anisotropy_flag'])
    prediction = prediction.argmax(0).astype(np.uint8)
    output = np.zeros(tuple(meta['original_shape']), dtype=np.uint8)
    a, b = meta['bbox']
    output[tuple(slice(int(x), int(y)) for x, y in zip(a, b))] = prediction
    segmentation = nib.Nifti1Image(output, meta['output_affine'])
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(segmentation.affine), nib.orientations.io_orientation(affine))
    segmentation = segmentation.as_reoriented(transform)
    return keep_largest_component(segmentation)


def compare_labels(reference, actual):
    a, b = np.asanyarray(reference.dataobj), np.asanyarray(actual.dataobj)
    if a.shape != b.shape or not np.allclose(reference.affine, actual.affine, atol=1e-5):
        raise RuntimeError('Segmentation grid differs from reference')
    dice = {}
    for code in range(1, 20):
        aa, bb = a == code, b == code
        total = int(aa.sum() + bb.sum())
        dice[str(code)] = 2 * int((aa & bb).sum()) / total if total else 1.0
    return {'voxel_agreement': float(np.mean(a == b)), 'dice': dice,
            'counts': np.bincount(b.ravel().astype(int), minlength=20).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--example', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(args.pipeline)
    torch.set_num_threads(8)
    rng = np.random.default_rng(42)
    phantom_results = []
    for name, shape, affine in [
        ('isotropic', (24, 28, 20), np.diag([1., 1., 1., 1.])),
        ('isotropic-flipped', (24, 28, 20), np.diag([-1., 1., -1., 1.])),
        ('resampled', (24, 28, 20), np.diag([1.2, 1.7, 2., 1.])),
        ('anisotropic', (24, 28, 8), np.diag([-1., 1., 4., 1.])),
        ('permuted', (24, 28, 20), np.array([[0., -1.5, 0., 5.], [1., 0., 0., 6.], [0., 0., 2., 7.], [0., 0., 0., 1.]])),
    ]:
        data = np.zeros(shape, dtype=np.float32)
        data[2:-2, 3:-3, 1:-1] = rng.uniform(1, 300, size=data[2:-2, 3:-3, 1:-1].shape)
        image = nib.Nifti1Image(data, affine); image.header.set_xyzt_units('mm')
        target = args.out / f'phantom-{name}.nii'; nib.save(image, target)
        meta = get_transforms()({'image': str(target)})
        info = json.loads(pipeline.preprocess(image.to_bytes()))
        pads = pipeline._context['pads']
        slices = tuple(slice(p[0], p[0] + n) for p, n in zip(pads, info['workingShape']))
        working = pipeline.prepared_image[slices]
        reference = np.asarray(meta['image'])[0]
        error = float(np.max(np.abs(reference - working)))
        if not np.allclose(reference, working, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f'Preprocessing differs for {name}: {error}')
        grid = np.indices(working.shape)
        labels = ((grid[0] // 4 + grid[1] // 5 + grid[2] // 3) % 4).astype(np.uint8)
        padded = np.pad(labels, pads)
        pipeline.recover(padded)
        actual = nib.Nifti1Image.from_bytes(pipeline.segmentation_bytes.tobytes())
        expected = original_recovery(labels, meta, affine)
        report = compare_labels(expected, actual)
        if report['voxel_agreement'] != 1.0:
            raise RuntimeError(f'Recovery differs for {name}')
        phantom_results.append({'name': name, 'preprocess_max_abs': error, 'recovery_agreement': 1.0})
    # Invalid files, volumes and values must be rejected without inference.
    rejected = []
    for name, data in [('empty', np.zeros((4, 4, 4), np.float32)), ('constant', np.ones((4, 4, 4), np.float32)), ('nonfinite', np.full((4, 4, 4), np.nan, np.float32)), ('4d', np.ones((4, 4, 4, 2), np.float32))]:
        try:
            pipeline.preprocess(nib.Nifti1Image(data, np.eye(4)).to_bytes())
        except ValueError:
            rejected.append(name)
        else:
            raise RuntimeError(f'Invalid input accepted: {name}')
    meta = get_transforms()({'image': str(args.example)})
    info = json.loads(pipeline.preprocess(args.example.read_bytes()))
    pads = pipeline._context['pads']
    slices = tuple(slice(p[0], p[0] + n) for p, n in zip(pads, info['workingShape']))
    error = float(np.max(np.abs(np.asarray(meta['image'])[0] - pipeline.prepared_image[slices])))
    if error > 2e-5:
        raise RuntimeError(f'Public template preprocessing differs: {error}')
    checkpoint = hf_hub_download(HF_REPO_ID, HF_MODEL_FILENAME, revision=HF_MODEL_REVISION)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    network = get_network(device, checkpoint_path=checkpoint).eval()
    began = time.perf_counter()
    with torch.inference_mode():
        logits = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=.5, mode='gaussian')(meta['image'].unsqueeze(0).to(device), network)
        labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        del logits
    expected = original_recovery(labels, meta, nib.load(args.example).affine)
    nib.save(expected, args.out / 'mni-pytorch-reference.nii.gz')
    reference_seconds = time.perf_counter() - began
    print(json.dumps({'phantoms': phantom_results, 'invalid_rejected': rejected, 'mni_preprocess_max_abs': error, 'pytorch_seconds': reference_seconds, 'device': device}), flush=True)
    # Native ONNX uses the same depth-major traversal as the bounded JS ring.
    options = ort.SessionOptions(); options.intra_op_num_threads = 8; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'], sess_options=options)
    shape = tuple(info['paddedShape'])
    stride = tuple(p // 2 for p in PATCH_SIZE)
    patches = sorted(dense_patch_slices(shape, PATCH_SIZE, stride), key=lambda s: (s[2].start, s[0].start, s[1].start))
    scores = np.zeros((20, *shape), np.float32); counts = np.zeros(shape, np.float32)
    importance = compute_importance_map(PATCH_SIZE, mode='gaussian', sigma_scale=.125).numpy()
    began = time.perf_counter()
    for index, region in enumerate(patches):
        patch = np.ascontiguousarray(pipeline.prepared_image[region])[None, None]
        output = session.run(None, {'image': patch})[0][0]
        scores[(slice(None), *region)] += output * importance
        counts[region] += importance
        print(f'Native ONNX public MNI patch {index + 1}/{len(patches)}', flush=True)
    predicted = (scores / counts).argmax(0).astype(np.uint8)
    del scores, counts
    pipeline.recover(predicted)
    actual = nib.Nifti1Image.from_bytes(pipeline.segmentation_bytes.tobytes())
    nib.save(actual, args.out / 'mni-onnx-reference.nii.gz')
    comparison = compare_labels(expected, actual)
    if comparison['voxel_agreement'] < .999 or any(value < .995 for value in comparison['dice'].values()):
        raise RuntimeError(f'Full-template ONNX labels differ materially: {comparison}')
    report = {'phantoms': phantom_results, 'invalid_rejected': rejected, 'mni': {**comparison, 'preprocess_max_abs': error, 'patches': len(patches), 'pytorch_seconds': reference_seconds, 'onnx_seconds': time.perf_counter() - began, 'pytorch_device': device}, 'scope': 'Public MNI template and generated phantoms; not clinical validation.'}
    (args.out / 'scientific-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
