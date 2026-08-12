import os
import sys
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

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
        return [inp], [out]

    # --- directory mode -----------------------------------------------------
    if os.path.isdir(inp):
        if os.path.isfile(out):
            sys.exit(
                f"Error: input is a directory but output ({out}) is an existing file.\n"
                "When -i is a directory, -o must be a directory."
            )
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            sys.exit(f"Error: cannot create output directory {out}: {e}")

        ignore_scans = set()
        if resume:
            print('Resuming: skipping scans already segmented in the output directory.')
            ignore_scans = {
                f.split('_seg.')[0] for f in os.listdir(out) if '_seg.' in f
            }

        inp_files, out_files = [], []
        skipped = []
        for f in sorted(os.listdir(inp)):
            full = os.path.join(inp, f)
            if not os.path.isfile(full) or not is_nifti(f):
                skipped.append(f)
                continue
            stem, ext = nifti_stem(f)
            if stem in ignore_scans:
                continue
            inp_files.append(full)
            out_files.append(os.path.join(out, f"{stem}_seg.{ext}"))

        if skipped:
            print(f"Note: ignoring {len(skipped)} non-NIfTI entr{'y' if len(skipped)==1 else 'ies'} "
                  f"in the input directory: {', '.join(skipped[:5])}"
                  + (' ...' if len(skipped) > 5 else ''))
        if not inp_files and not resume:
            sys.exit(f"Error: no NIfTI files (.nii / .nii.gz) found in directory: {inp}")
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
    1) --model_path CLI argument
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
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        )
    except Exception as e:
        sys.exit(
            "Error: could not download the MindGlide model weights (~123 MB) from the\n"
            f"Hugging Face Hub ({HF_REPO_ID}). If you are offline, download the file\n"
            f"once from https://huggingface.co/{HF_REPO_ID} and pass it with --model_path.\n"
            f"Reason: {e}"
        )


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
        "--model_path", "--model-path",
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
        '--sw_batch_size', '--sw-batch-size',
        dest='sw_batch_size',
        type=int,
        default=4,
        help="Sliding-window batch size (default: 4). Lower this if you run out of GPU memory."
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Skip scans that have already been segmented in the output directory."
    )
    parser.add_argument(
        '--no_klc', '--no-klc',
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
    args = parse_args()

    print(CITATION)

    # Validate I/O before loading the heavy libraries so user errors fail fast.
    inp_files, out_files = collect_io(args.i, args.o, resume=args.resume)

    if len(inp_files) == 0:
        print('Found 0 new images to segment. Exiting.')
        return

    print(f"Found {len(inp_files)} image{'s' if len(inp_files) != 1 else ''} to process.")

    import numpy as np
    import nibabel as nib
    import torch
    from tqdm import tqdm

    from monai.inferers import SlidingWindowInferer
    from monai.data import Dataset, DataLoader
    from monai.transforms import AsDiscrete

    from mindglide.network import get_network
    from mindglide.transforms import get_transforms, recovery_prediction, keep_largest_component
    from mindglide.consts import PATCH_SIZE, PROPERTIES

    DEVICE = resolve_device(args.device)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("Tip: CPU inference typically takes a few minutes per scan. "
              "A CUDA GPU runs in seconds.")

    num_classes = len(PROPERTIES['labels'])
    as_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)

    # ===============================================
    # Download and initialise the model.
    # ===============================================
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
    # Prepare the datasets.
    # ===============================================

    # Cheap preflight (header read only): report unreadable files one by one
    # instead of crashing the whole run inside a DataLoader worker.
    n_ok, failed = 0, []
    readable = []
    for f, o in zip(inp_files, out_files):
        try:
            nib.load(f)
            readable.append((f, o))
        except Exception as e:
            failed.append(f)
            print(f"⚠️ Skipping unreadable NIfTI file: {f}\nReason: {e}")

    # convert for MONAI dataset class formatting
    data = [{'image': f, 'output': o} for f, o in readable]

    # Create MONAI dataset and dataloader
    # The transforms handle preprocessing like resizing and intensity normalization
    dataset = Dataset(data=data, transform=get_transforms(no_reorient=args.no_reorient))
    num_workers = min(4, len(data), os.cpu_count() or 1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # ===============================================
    # Run the inference script
    # ===============================================

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Segmenting Images"):
            scan_name = str(batch['image_meta_dict']['filename_or_obj'][0])
            try:
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

                    # convert the prediction into [K, H, W, D] where K is
                    # the number of anatomical tissues.
                    pred = as_discrete(predictions[idx])

                    # the input scan is resampled if it's anisotropic. In this
                    # case, we need to transform the segmentation back to the input
                    # space. To do this, we need some metadata that have been stored
                    # by the `PreprocessAnisotropic` transform.
                    resample_flag       = batch["resample_flag"][idx].item()
                    anisotrophy_flag    = batch["anisotrophy_flag"][idx].item()
                    crop_shape          = batch["crop_shape"][idx].tolist()
                    original_shape      = batch["original_shape"][idx].tolist()
                    bbox                = batch["bbox"][idx].tolist()

                    if resample_flag:
                        pred = recovery_prediction(pred, [num_classes, *crop_shape], anisotrophy_flag)

                    # Finally, select the class of highest probability and create a
                    # segmentation map (H, W, D) where [i,j,k] indicates the anatomical
                    # label of the voxel at that position.
                    pred = np.argmax(pred, axis=0)

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
                        back_to_orig_ornt = nib.orientations.ornt_transform(current_orientation, original_orientation)
                        nifti_img = nifti_img.as_reoriented(back_to_orig_ornt)

                    # Keep the largest component of the segmentation (removes small regions
                    # outside of the brain).
                    if not args.no_klc:
                        nifti_img = keep_largest_component(nifti_img)

                    # Save the output.
                    nib.save(nifti_img, opaths[idx])
                    n_ok += 1

            except Exception as e:
                failed.append(scan_name)
                print(f"\n⚠️ Error processing scan: {scan_name}")
                print(f"Reason: {e}")
                if DEVICE.type == 'cuda' and 'out of memory' in str(e).lower():
                    print("Hint: reduce --sw_batch_size (e.g. --sw_batch_size 1) "
                          "or run with --device cpu.")
                if DEVICE.type == 'mps':
                    print("Hint: some 3D operations are not supported on Apple MPS. "
                          "Try --device cpu, or set PYTORCH_ENABLE_MPS_FALLBACK=1.")
                continue

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
