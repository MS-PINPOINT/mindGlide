"""MindGlide preprocessing/recovery running inside browser WebAssembly.

No filesystem, network, telemetry, or patient identifiers are used here.
NumPy/SciPy/scikit-image preserve the reference interpolation semantics.
"""
import gc
import gzip
import io
import json

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize

PATCH = (128, 128, 64)
MAX_DECOMPRESSED = 256 * 1024 * 1024
MAX_ORIGINAL_VOXELS = 50_000_000
MAX_RESAMPLED_VOXELS = 20_000_000
MAX_ACCUMULATOR_BYTES = 512 * 1024 * 1024
_context = {}
prepared_image = None
segmentation_bytes = None


def _resize(image, shape, order, mode):
    return resize(image, shape, order=order, mode=mode, cval=0,
                  clip=True, anti_aliasing=False)


def preprocess(raw):
    global _context, prepared_image
    clear()
    raw = bytes(raw)
    if raw[:2] == b'\x1f\x8b':
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(MAX_DECOMPRESSED + 1)
    if len(raw) > MAX_DECOMPRESSED:
        raise ValueError('The decompressed scan exceeds the browser size limit. Use the local Python tool.')
    if len(raw) < 352:
        raise ValueError('This file does not contain a complete NIfTI header.')
    header_size = int.from_bytes(raw[:4], 'little')
    if header_size not in (348, 540):
        header_size = int.from_bytes(raw[:4], 'big')
    image_type = {348: nib.Nifti1Image, 540: nib.Nifti2Image}.get(header_size)
    if image_type is None:
        raise ValueError('Choose a NIfTI brain volume (.nii or .nii.gz).')
    image = image_type.from_bytes(raw)
    if len(image.shape) != 3 or any(x < 2 or x > 1024 for x in image.shape):
        raise ValueError('Choose a single three-dimensional MRI volume.')
    if int(np.prod(image.shape)) > MAX_ORIGINAL_VOXELS:
        raise ValueError('This scan is too large for this browser release. Use the local Python tool.')
    units = image.header.get_xyzt_units()[0]
    if units not in ('mm', 'unknown'):
        raise ValueError('This release requires NIfTI spatial units in millimetres.')
    affine = image.affine.copy()
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError('The scan has an invalid spatial transform.')
    header_spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    affine_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    if (not np.isfinite(header_spacing).all() or np.any(header_spacing <= 0)
            or not np.allclose(header_spacing, affine_spacing, rtol=1e-3, atol=1e-4)):
        raise ValueError('The scan has inconsistent or invalid voxel spacing. Check its NIfTI header.')
    canonical = nib.as_closest_canonical(image)
    spacing = np.sqrt(np.sum(canonical.affine[:3, :3] ** 2, axis=0))
    if not np.isfinite(spacing).all() or np.any(spacing < 0.1) or np.any(spacing > 20):
        raise ValueError('The scan has unsupported or invalid voxel spacing.')
    array = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError('The scan contains non-finite intensity values.')
    foreground = array > 0  # MONAI generate_spatial_bounding_box default.
    starts, ends = [], []
    for axis in range(3):
        positions = np.flatnonzero(foreground.any(axis=tuple(i for i in range(3) if i != axis)))
        if not len(positions):
            raise ValueError('No MRI foreground was found. Check that this is a brain scan.')
        starts.append(int(positions[0]))
        ends.append(int(positions[-1]) + 1)
    del foreground
    cropped = array[tuple(slice(a, b) for a, b in zip(starts, ends))]
    crop_shape = cropped.shape
    resample = not np.array_equal(spacing, np.ones(3))
    anisotropy = bool(np.max(spacing) / np.min(spacing) >= 3)
    new_shape = tuple((spacing * np.asarray(crop_shape)).astype(int)) if resample else crop_shape
    if any(x < 1 for x in new_shape) or int(np.prod(new_shape)) > MAX_RESAMPLED_VOXELS:
        raise ValueError('The 1 mm working volume exceeds this browser release’s size limit. Use the local Python tool.')
    padded_shape = tuple(max(p, s) for p, s in zip(PATCH, new_shape))
    if int(np.prod(padded_shape)) > MAX_RESAMPLED_VOXELS:
        raise ValueError('The padded working volume needs too much browser memory. Use the local Python tool.')
    if resample and anisotropy and new_shape[0] * new_shape[1] * crop_shape[2] > MAX_RESAMPLED_VOXELS:
        raise ValueError('Resampling this scan needs too much browser memory. Use the local Python tool.')
    slab_bytes = 21 * padded_shape[0] * padded_shape[1] * PATCH[2] * 4
    if slab_bytes > MAX_ACCUMULATOR_BYTES:
        raise ValueError('The working volume needs too much browser memory. Use the local Python tool.')
    if resample:
        if anisotropy:
            slices = [_resize(cropped[:, :, z], new_shape[:2], 3, 'edge')
                      for z in range(crop_shape[2])]
            working = _resize(np.stack(slices, axis=-1), new_shape, 0, 'constant')
        else:
            working = _resize(cropped, new_shape, 3, 'edge')
    else:
        working = cropped.copy()
    nonzero = working != 0
    values = working[nonzero]
    mean = np.mean(values, dtype=np.float32)
    std = np.std(values, dtype=np.float32)
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError('The scan has no usable intensity variation.')
    working[nonzero] = (values - mean) / std
    pads = tuple(((p - s) // 2, p - s - (p - s) // 2)
                 for p, s in zip(padded_shape, new_shape))
    prepared_image = np.ascontiguousarray(np.pad(working, pads), dtype=np.float32)
    _context = {'image': image, 'affine': affine, 'canonical_affine': canonical.affine.copy(),
                'original_shape': canonical.shape, 'crop_shape': crop_shape,
                'starts': starts, 'ends': ends, 'working_shape': new_shape,
                'padded_shape': padded_shape, 'pads': pads, 'resample': resample,
                'anisotropy': anisotropy, 'units': units}
    canonical.uncache()
    image.uncache()
    info = {'shape': [int(x) for x in image.shape], 'spacing': [float(x) for x in image.header.get_zooms()[:3]],
            'workingShape': [int(x) for x in new_shape], 'paddedShape': [int(x) for x in padded_shape],
            'slabBytes': int(slab_bytes), 'unitsAssumed': units == 'unknown'}
    return json.dumps(info)


def recover(labels):
    global segmentation_bytes
    context = _context
    labels = np.asarray(labels, dtype=np.uint8).reshape(context['padded_shape'])
    labels = labels[tuple(slice(pad[0], pad[0] + s)
                          for pad, s in zip(context['pads'], context['working_shape']))]
    if context['resample']:
        # Same thresholds and lowest-label tie rule as recovery_prediction's
        # one-hot + argmax, while avoiding a full 20-channel recovery tensor.
        recovered = np.zeros(context['crop_shape'], dtype=np.uint8)
        for code in range(1, 20):
            mask = (labels == code).astype(float)
            if context['anisotropy']:
                depth = context['crop_shape'][2]
                along_depth = _resize(mask, (*mask.shape[:2], depth), 0, 'constant') >= 0.5
                for z in range(depth):
                    plane = _resize(along_depth[:, :, z].astype(float),
                                    context['crop_shape'][:2], 1, 'edge') >= 0.5
                    target = recovered[:, :, z]
                    target[(target == 0) & plane] = code
            else:
                mask = _resize(mask, context['crop_shape'], 1, 'edge') >= 0.5
                recovered[(recovered == 0) & mask] = code
        labels = recovered
    padded = np.zeros(context['original_shape'], dtype=np.uint8)
    padded[tuple(slice(a, b) for a, b in zip(context['starts'], context['ends']))] = labels
    segmentation = nib.Nifti1Image(padded, context['canonical_affine'])
    current = nib.orientations.io_orientation(segmentation.affine)
    original = nib.orientations.io_orientation(context['affine'])
    if not np.array_equal(current, original):
        segmentation = segmentation.as_reoriented(nib.orientations.ornt_transform(current, original))
    data = np.asanyarray(segmentation.dataobj).copy()
    components, count = ndimage.label(data > 0)  # 6-connectivity, as in SciPy reference.
    if count:
        sizes = np.bincount(components.ravel())[1:]
        data[components != int(sizes.argmax()) + 1] = 0
    segmentation = nib.Nifti1Image(data, segmentation.affine)
    segmentation.header.set_xyzt_units('mm')
    voxel_volume = float(np.prod(context['image'].header.get_zooms()[:3]))
    counts = np.bincount(data.ravel(), minlength=20)
    segmentation_bytes = np.frombuffer(segmentation.to_bytes(), dtype=np.uint8).copy()
    del components
    gc.collect()
    return json.dumps({'volumes': [float(n) * voxel_volume for n in counts],
                       'counts': [int(n) for n in counts],
                       'empty': not bool(np.any(data))})


def clear():
    global _context, prepared_image, segmentation_bytes
    if prepared_image is not None:
        prepared_image.fill(0)
    if segmentation_bytes is not None:
        segmentation_bytes.fill(0)
    _context = {}
    prepared_image = None
    segmentation_bytes = None
    gc.collect()
