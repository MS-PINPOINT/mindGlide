"""MindGlide: brain MRI segmentation for multiple sclerosis — any modality, any quality.

Command-line entry points: ``mindglide`` (segmentation) and ``mindglide-volumes``
(per-region volumes). See https://github.com/MS-PINPOINT/mindGlide.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mindglide")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
