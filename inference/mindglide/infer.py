import argparse
import os
import sys
import warnings
from pathlib import Path

CITATION = """
If you use this tool, please cite the original MindGlide paper:
------
Goebl, P., Wingrove, J., Abdelmannan, O., Brito Vega, B., Stutters,
J., Ramos, S. D. G., ... & Eshaghi, A. (2025).
Enabling new insights from old scans by repurposing clinical MRI archives for multiple sclerosis research.
Nature Communications, 16(1), 3149.
------
"""

HF_REPO_ID = "MS-PINPOINT/mindglide"
HF_MODEL_FILENAME = "_20240404_conjurer_trained_dice_7733.pt"
# Pin the exact model-repo commit so a fresh install always downloads the same
# weights, even if the Hugging Face repo's main branch moves. Bump deliberately
# when releasing new weights.
HF_MODEL_REVISION = "a1969821c0a4a37ae54f649a9a0c6fd1b8a48e26"

NIFTI_SUFFIXES = ('.nii', '.nii.gz')


def get_version():
    try:
        from importlib.metadata import version
        return version("mindglide")
    except Exception:
        return "unknown"


def is_nifti(path):
    return str(path).lower().endswith(NIFTI_SUFFIXES)


def nifti_stem(filename):
    """'sub-01_T1w.nii.gz' -> ('sub-01_T1w', 'nii.gz')"""
    name = str(filename)
    for suffix in NIFTI_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)], suffix.lstrip('.')
    return name, ''


def cuda_is_usable():
    """
    True only if a CUDA device exists AND can actually run kernels.
    torch.cuda.is_available() alone is not enough: pip's default PyTorch wheels
    drop older GPU architectures (e.g. Pascal cards like the GTX 10xx / Quadro
    P series with current cu12x wheels), in which case every kernel launch
    fails with 'no kernel image is available' even though CUDA is 'available'.
    """
    import torch
    if not torch.cuda.is_available():
        return False, "no CUDA GPU detected by PyTorch"
    try:
        (torch.zeros(1, device="cuda") + 1).item()
        return True, None
    except RuntimeError as e:
        gpu = torch.cuda.get_device_name(0)
        return False, (
            f"your GPU ({gpu}) is detected but this PyTorch build cannot run on it.\n"
            f"Underlying error: {str(e).splitlines()[0]}\n"
            "This usually means the pre-built PyTorch wheels no longer include kernels\n"
            "for your GPU's architecture. Install a PyTorch build that supports it\n"
            "(see https://pytorch.org/get-started/locally/ — an older CUDA variant,\n"
            "e.g. the cu118 wheels, often restores support for older GPUs)."
        )


def resolve_device(choice="auto"):
    """
    Resolve the compute device. With "auto", pick the best available:
    working CUDA (GPU) > MPS (Apple Silicon) > CPU.
    """
    import torch

    def mps_available():
        try:
            return torch.backends.mps.is_available()
        except AttributeError:
            return False

    if choice == "auto":
        if torch.cuda.is_available():
            usable, reason = cuda_is_usable()
            if usable:
                return torch.device("cuda")
            print(f"Warning: not using the GPU — {reason}")
            print("Falling back to CPU.\n")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        usable, reason = cuda_is_usable()
        if not usable:
            sys.exit(f"Error: --device cuda requested but {reason}\n"
                     "Run with --device cpu to use the CPU instead.")
    if choice == "mps" and not mps_available():
        sys.exit("Error: --device mps requested but MPS (Apple Silicon) is not available. "
                 "Run with --device cpu instead.")
    return torch.device(choice)


def _check_no_overwrite(inp_files, out_files):
    """Refuse any plan in which an output path would overwrite an input scan."""
    inputs = {os.path.realpath(f) for f in inp_files}
    clobbered = [o for o in out_files if os.path.realpath(o) in inputs]
    if clobbered:
        listing = '\n'.join(f"  - {c}" for c in clobbered)
        sys.exit(
            "Error: refusing to overwrite input scan(s) with segmentation output:\n"
            f"{listing}\n"
            "Choose a different output path."
        )


def collect_io(inp, out, resume=False):
    """
    Validate input/output paths and return the list of (input, output) file
    pairs to process. Exits with a clear message on user error.
    """
    inp, out = str(inp), str(out)

    # --- single-file mode ---------------------------------------------------
    if is_nifti(inp):
        if not os.path.isfile(inp):
            sys.exit(f"Error: input file not found: {inp}")
        if os.path.isdir(out):
            # allow "-i scan.nii.gz -o existing_dir/" for convenience
            stem, ext = nifti_stem(os.path.basename(inp))
            out = os.path.join(out, f"{stem}_seg.{ext}")
        elif not is_nifti(out):
            sys.exit(
                f"Error: output path must end in .nii or .nii.gz (got: {out}).\n"
                "Example: mindglide -i scan.nii.gz -o scan_seg.nii.gz"
            )
        parent = os.path.dirname(os.path.abspath(out))
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            sys.exit(f"Error: cannot create output directory {parent}: {e}")
        if resume and os.path.exists(out):
            print(f"Resuming: output already exists, skipping ({out}).")
            return [], []
        _check_no_overwrite([inp], [out])
        return [inp], [out]

    # --- directory mode -----------------------------------------------------
    if os.path.isdir(inp):
        if os.path.isfile(out):
            sys.exit(
                f"Error: input is a directory but output ({out}) is an existing file.\n"
                "When -i is a directory, -o must be a directory."
            )
        if is_nifti(out) and not os.path.isdir(out):
            sys.exit(
                f"Error: -i is a directory, so -o must be an output directory\n"
                f"(got what looks like a NIfTI filename: {out})."
            )
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            sys.exit(f"Error: cannot create output directory {out}: {e}")

        if resume:
            print('Resuming: skipping scans whose segmentation already exists '
                  'in the output directory.')

        inp_files, out_files = [], []
        skipped, seg_inputs = [], []
        n_niftis = 0
        for f in sorted(os.listdir(inp)):
            full = os.path.join(inp, f)
            if not os.path.isfile(full) or not is_nifti(f):
                skipped.append(f)
                continue
            n_niftis += 1
            stem, ext = nifti_stem(f)
            if stem.endswith('_seg'):
                # Almost certainly a previous MindGlide output (e.g. when the
                # input and output directories are the same); segmenting a
                # segmentation is never useful.
                seg_inputs.append(f)
                continue
            if resume and os.path.exists(os.path.join(out, f"{stem}_seg.{ext}")):
                continue
            inp_files.append(full)
            out_files.append(os.path.join(out, f"{stem}_seg.{ext}"))

        if skipped:
            print(f"Note: ignoring {len(skipped)} non-NIfTI entr{'y' if len(skipped)==1 else 'ies'} "
                  f"in the input directory: {', '.join(skipped[:5])}"
                  + (' ...' if len(skipped) > 5 else ''))
        if seg_inputs:
            print(f"Note: ignoring {len(seg_inputs)} file{'s' if len(seg_inputs) != 1 else ''} "
                  f"that look like previous segmentations (*_seg): {', '.join(seg_inputs[:5])}"
                  + (' ...' if len(seg_inputs) > 5 else ''))
        if n_niftis == 0:
            sys.exit(f"Error: no NIfTI files (.nii / .nii.gz) found in directory: {inp}")

        # Distinct inputs (e.g. scan.nii vs SCAN.nii.gz) must never share an output.
        dupes = sorted({o for o in out_files if out_files.count(o) > 1})
        if dupes:
            listing = '\n'.join(f"  - {d}" for d in dupes)
            sys.exit(f"Error: multiple input files map to the same output file:\n{listing}\n"
                     "Rename the conflicting inputs.")

        _check_no_overwrite(inp_files, out_files)
        return inp_files, out_files

    # --- neither ------------------------------------------------------------
    if not os.path.exists(inp):
        sys.exit(f"Error: input path not found: {inp}")
    sys.exit(
        f"Error: input must be a NIfTI file (.nii / .nii.gz) or a directory of NIfTI files (got: {inp})."
    )


def resolve_model_path(cli_model_path=None):
    """
    Resolve the model checkpoint. Priority:
    1) --model-path CLI argument
    2) MODEL_PATH environment variable
    3) automatic download from the Hugging Face Hub (cached after first run)
    """
    if cli_model_path is not None:
        path = Path(cli_model_path)
        if not path.is_file():
            sys.exit(f"Error: model checkpoint not found: {path}")
        return path

    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        path = Path(env_model_path)
        if path.is_file():
            return path
        print(f"Warning: MODEL_PATH is set but does not exist ({env_model_path}); "
              "falling back to the Hugging Face download.")

    from huggingface_hub import hf_hub_download
    try:
        return Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILENAME,
                revision=HF_MODEL_REVISION,
            )
        )
    except Exception as e:
        sys.exit(
            "Error: could not download the MindGlide model weights (~123 MB) from the\n"
            f"Hugging Face Hub ({HF_REPO_ID}). If you are offline, download the file\n"
            f"once from https://huggingface.co/{HF_REPO_ID} and pass it with --model-path.\n"
            f"Reason: {e}"
        )


def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {value})")
    return ivalue


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="mindglide",
        description="MindGlide: brain MRI segmentation for multiple sclerosis "
                    "(any modality, any quality). Works on GPU or CPU.",
        epilog="Examples:\n"
               "  mindglide -i scan.nii.gz -o scan_seg.nii.gz\n"
               "  mindglide -i scans_dir/ -o segmentations_dir/\n"
               "  mindglide -i scan.nii.gz -o out.nii.gz --device cpu\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-i',
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a NIfTI file (.nii / .nii.gz) or a directory of NIfTI images."
    )
    parser.add_argument(
        '-o',
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the output NIfTI file, or output directory when -i is a directory."
    )
    parser.add_argument(
        "--model-path", "--model_path",
        dest="model_path",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a local .pt checkpoint. If set, skips the Hugging Face download."
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help="Compute device (default: auto — picks GPU if available, else CPU)."
    )
    parser.add_argument(
        '--sw-batch-size', '--sw_batch_size',
        dest='sw_batch_size',
        type=positive_int,
        default=4,
        help="Sliding-window batch size (default: 4). Lower this if you run out of GPU memory."
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Skip scans whose segmentation already exists at the output location."
    )
    parser.add_argument(
        '--no-klc', '--no_klc',
        dest='no_klc',
        action='store_true',
        help="Disable 'Keep Largest Component' post-processing."
    )
    parser.add_argument(
        '--no-reorient', '--no_reorient',
        dest='no_reorient',
        action='store_true',
        help=(
            "Disable automatic re-orientation to RAS coordinates before inference. "
            "Note: The final output will always be aligned with the original input "
            "scan, regardless of this setting."
        )
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f"mindglide {get_version()}"
    )
    return parser.parse_args(argv)


def main():
    """
    Runs the MindGlide model inference on NIfTI file(s).
    """
    # CLI-only: keep terminal output clean. Deliberately inside main() so that
    # importing mindglide as a library does not silence the host's warnings.
    warnings.filterwarnings("ignore")

    args = parse_args()

    print(CITATION)

    # Validate paths before loading the heavy libraries so user errors fail fast.
    inp_files, out_files = collect_io(args.i, args.o, resume=args.resume)
    if args.model_path is not None and not Path(args.model_path).is_file():
        sys.exit(f"Error: model checkpoint not found: {args.model_path}")

    if len(inp_files) == 0:
        print('Found 0 new images to segment. Exiting.')
        return

    print(f"Found {len(inp_files)} image{'s' if len(inp_files) != 1 else ''} to process.")

    import nibabel as nib
    import numpy as np
    import torch
    from monai.data import DataLoader, Dataset
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import AsDiscrete
    from tqdm import tqdm

    from mindglide.consts import PATCH_SIZE, PROPERTIES
    from mindglide.network import get_network
    from mindglide.transforms import get_transforms, keep_largest_component, recovery_prediction

    DEVICE = resolve_device(args.device)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("Tip: CPU inference typically takes a few minutes per scan. "
              "A CUDA GPU runs in seconds.")

    num_classes = len(PROPERTIES['labels'])
    as_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)

    # Cheap preflight (header read only): report unreadable files one by one.
    # Runs before the model download so a bad batch fails fast.
    n_ok, failed = 0, []
    readable = []
    for f, o in zip(inp_files, out_files):
        try:
            nib.load(f)
            readable.append((f, o))
        except Exception as e:
            failed.append(f)
            print(f"Warning: skipping unreadable NIfTI file: {f}\nReason: {e}")

    # ===============================================
    # Download and initialise the model.
    # ===============================================
    if readable:
        model_path = resolve_model_path(args.model_path)

        # Instantiate MindGlide network and load weights
        net = get_network(checkpoint_path=model_path, device=DEVICE)
        net = net.eval()

        # Instantiate the sliding window inferer for memory-efficient processing
        patch_inferer = SlidingWindowInferer(
            roi_size=PATCH_SIZE,
            sw_batch_size=args.sw_batch_size,
            overlap=0.5,
            mode='gaussian',
        )

    # ===============================================
    # Prepare the dataset.
    # ===============================================

    # convert for MONAI dataset class formatting
    data = [{'image': f, 'output': o} for f, o in readable]

    # Create MONAI dataset and dataloader
    # The transforms handle preprocessing like resizing and intensity normalization
    dataset = Dataset(data=data, transform=get_transforms(no_reorient=args.no_reorient))
    num_workers = min(4, len(data), os.cpu_count() or 1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # ===============================================
    # Run the inference loop
    # ===============================================

    def save_atomically(nifti_img, out_path):
        # Write to a temp file in the same directory, then rename into place, so
        # an interrupted run can never leave a truncated *_seg file behind
        # (which --resume would otherwise treat as done). The temp name keeps
        # the full NIfTI suffix so nibabel still applies gzip compression.
        tmp_path = os.path.join(os.path.dirname(out_path) or '.',
                                f".tmp{os.getpid()}-{os.path.basename(out_path)}")
        try:
            nib.save(nifti_img, tmp_path)
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Batches arrive in dataset order (batch_size=1, shuffle=False), so `cursor`
    # always indexes the scan the next batch belongs to. Fetching is inside the
    # guard because a corrupt data payload with an intact header passes the
    # preflight and only fails inside the DataLoader worker.
    with torch.inference_mode():
        loader = iter(dataloader)
        cursor = 0
        progress = tqdm(total=len(data), desc="Segmenting Images")
        while cursor < len(data):
            scan_name = data[cursor]['image']
            try:
                try:
                    batch = next(loader)
                except StopIteration:
                    break

                images = batch['image'].to(DEVICE)
                opaths = batch['output']

                # Run sliding window inference
                predictions = patch_inferer(images, net).cpu()

                # Post-process and save each prediction in the batch
                for idx in range(predictions.shape[0]):

                    # The image is re-oriented to RAS as part of our set of pre-processing.
                    # The original orientation can be recovered from the original affine matrix,
                    # which is stored inside `image_meta_dict` (this is not affected by the
                    # transforms applied to the input).
                    original_affine = batch['image_meta_dict']['affine'][idx].numpy()
                    original_orientation = nib.orientations.io_orientation(original_affine)

                    # the input scan is resampled if it's anisotropic. In this
                    # case, we need to transform the segmentation back to the input
                    # space. To do this, we need some metadata that have been stored
                    # by the `PreprocessAnisotropic` transform.
                    resample_flag       = batch["resample_flag"][idx].item()
                    anisotropy_flag     = batch["anisotropy_flag"][idx].item()
                    crop_shape          = batch["crop_shape"][idx].tolist()
                    original_shape      = batch["original_shape"][idx].tolist()
                    bbox                = batch["bbox"][idx].tolist()

                    # Select the class of highest probability per voxel to build a
                    # segmentation map (H, W, D) where [i,j,k] is the anatomical
                    # label of that voxel. recovery_prediction needs a one-hot
                    # [K, H, W, D] volume, so only the resampled path pays for one.
                    if resample_flag:
                        pred = as_discrete(predictions[idx])
                        pred = recovery_prediction(pred, [num_classes, *crop_shape], anisotropy_flag)
                        pred = np.argmax(pred, axis=0)
                    else:
                        pred = predictions[idx].argmax(dim=0).numpy()

                    # This is still part of the recovery process to get the prediction
                    # to the input space. Specifically, we pad the cropped prediction back
                    # to the original image size.
                    pred_padded = np.zeros(original_shape, dtype=pred.dtype)
                    (h_start, w_start, d_start), (h_end, w_end, d_end) = bbox
                    pred_padded[h_start:h_end, w_start:w_end, d_start:d_end] = pred

                    # This is the correct affine of the segmentation (the affine of the input
                    # has been updated subject to different transformations, e.g., OrientationD).
                    current_affine = batch["output_affine"][idx]
                    nifti_img = nib.Nifti1Image(pred_padded.astype(np.uint8), current_affine)

                    # Move the segmentation back to the original orientation.
                    current_orientation = nib.orientations.io_orientation(current_affine)

                    if not np.all(current_orientation == original_orientation):
                        back_to_orig_ornt = nib.orientations.ornt_transform(
                            current_orientation, original_orientation)
                        nifti_img = nifti_img.as_reoriented(back_to_orig_ornt)

                    # Keep the largest component of the segmentation (removes small regions
                    # outside of the brain).
                    if not args.no_klc:
                        nifti_img = keep_largest_component(nifti_img)

                    save_atomically(nifti_img, opaths[idx])
                    n_ok += 1

            except Exception as e:
                failed.append(scan_name)
                print(f"\nError processing scan: {scan_name}")
                print(f"Reason: {e}")
                if DEVICE.type == 'cuda' and 'out of memory' in str(e).lower():
                    print("Hint: reduce --sw-batch-size (e.g. --sw-batch-size 1) "
                          "or run with --device cpu.")
                if DEVICE.type == 'mps':
                    print("Hint: some 3D operations are not supported on Apple MPS. "
                          "Try --device cpu, or set PYTORCH_ENABLE_MPS_FALLBACK=1.")

            cursor += 1
            progress.update(1)
        progress.close()

        # If the loader stopped early (e.g. a wedged worker pool), account for
        # every scan so the summary and exit code stay truthful.
        for entry in data[cursor:]:
            failed.append(entry['image'])
            print(f"\nError: scan was never processed (data loader stopped early): {entry['image']}")

    # ===============================================
    # Summarise
    # ===============================================
    if failed:
        print(f"\nFinished with errors: {n_ok} scan{'s' if n_ok != 1 else ''} segmented, "
              f"{len(failed)} failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)

    print(f"\nInference complete. {n_ok} segmentation{'s' if n_ok != 1 else ''} saved to: {args.o}")


if __name__ == "__main__":
    main()
