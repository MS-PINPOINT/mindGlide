"""MindGlide: brain MRI segmentation for multiple sclerosis — any sequence, any quality.

Command line:  ``mindglide -i scan.nii.gz -o scan_seg.nii.gz``  (and ``mindglide-volumes``).

Python:

    from mindglide import segment, volumes_dataframe
    seg_path = segment("scan.nii.gz")          # writes scan_seg.nii.gz
    df = volumes_dataframe(seg_path)           # per-region volumes in mm3

See https://github.com/MS-PINPOINT/mindGlide.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mindglide")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0+unknown"

from mindglide.infer import UsageError, segment

__all__ = ["__version__", "segment", "UsageError", "volumes_dataframe", "calculate_volumes"]

_LAZY = {"volumes_dataframe", "calculate_volumes"}


def __getattr__(name):
    # Lazy: importing mindglide stays fast; pandas/nibabel load only when the
    # volumes helpers are actually used.
    if name in _LAZY:
        from mindglide import volumes

        return getattr(volumes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
