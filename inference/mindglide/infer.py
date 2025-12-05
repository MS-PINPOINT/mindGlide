import os
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")


def get_best_device():
    """
    Select the best available device for computation.
    Priority: CUDA (GPU) > MPS (Apple Silicon) > CPU.
    """
    import torch
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.mps.is_available():  return torch.device("mps")
    return torch.device("cpu")


def main():
    """
    Runs the MindGlide model inference on a directory of NIfTI files.
    """
    parser = argparse.ArgumentParser(description="MindGlide: Brain Segmentation Inference Tool")

    parser.add_argument(
        '-i',
        type=str, 
        required=True, 
        metavar="PATH",
        help="Path to a NIfTI file or a directory containing NIfTI images."
    )

    parser.add_argument(
        '-o', 
        type=str, 
        required=True, 
        metavar="PATH",
        help="Path to the output NIfTI file or directory."
    )

    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
        metavar="FILE",
        help="Path to local .pt checkpoint. If set, skips HuggingFace download."
    )

    parser.add_argument(
        '--sw_batch_size', 
        type=int, 
        default=4, 
        help="Batch size for the sliding window inferer."
    )

    parser.add_argument(
        '--resume', 
        action='store_true', 
        help="Skip scans that have already been segmented in the output directory."
    )

    parser.add_argument(
        '--no_klc', 
        action='store_true', 
        help="Disable 'Keep Largest Component' post-processing."
    )

    parser.add_argument(
        '--no-reorient', 
        action='store_true', 
        help=(
            "Disable automatic re-orientation to RAS coordinates before inference. "
            "Note: The final output will always be aligned with the original input "
            "scan, regardless of this setting."
        )
    )

    args = parser.parse_args()

    print("""
If you use this tool, please cite the original MindGlide paper:
------
Goebl, P., Wingrove, J., Abdelmannan, O., Brito Vega, B., Stutters, 
J., Ramos, S. D. G., ... & Eshaghi, A. (2025). 
Enabling new insights from old scans by repurposing clinical MRI archives for multiple sclerosis research. 
Nature Communications, 16(1), 3149.
------
    """)

    # Load libraries only when the input is validated.
    import numpy as np
    import nibabel as nib
    import torch
    from tqdm import tqdm
    from huggingface_hub import hf_hub_download

    from monai.inferers import SlidingWindowInferer
    from monai.data import Dataset, DataLoader
    from monai.transforms import AsDiscrete

    from mindglide.network import get_network
    from mindglide.transforms import get_transforms, recovery_prediction, keep_largest_component
    from mindglide.consts import PATCH_SIZE, PROPERTIES

    DEVICE = get_best_device()
    print(f"Using device: {DEVICE}")

    num_classes = len(PROPERTIES['labels'])
    as_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    

    # ===============================================
    # Parse I/O
    # ===============================================
    inp, out = args.i, args.o

    def is_nifti(path):
        return path.endswith(('.nii', '.nii.gz'))

    if is_nifti(inp) and is_nifti(out):
        inp_files = [inp]
        out_files = [out]

    elif os.path.isdir(inp):
        os.makedirs(out, exist_ok=True)

        ignore_scans = set()
        if args.resume:
            print('Ignoring scans already segmented.')
            ignore_scans = {
                f.split('_seg.')[0] for f in os.listdir(out) if '_seg.' in f
            }

        inp_files, out_files = [], []
        for f in os.listdir(inp):
            name, ext = f.split('.', 1)
            if name in ignore_scans or not is_nifti(f):
                continue
            inp_files.append(os.path.join(inp, f))
            out_files.append(os.path.join(out, f"{name}_seg.{ext}"))

    else:
        print("Error: invalid input/output paths.")
        exit(1)

    # ===============================================
    # Download and initialise the model.
    # ===============================================
    # Download the weights from HF / resolve checkpoint path
    env_model_path = os.getenv("MODEL_PATH")

    if env_model_path is not None and Path(env_model_path).is_file():
        # 1) Prefer MODEL_PATH if set and exists
        model_path = Path(env_model_path)
    elif args.model_path is not None:
        # 2) Then fall back to CLI argument
        model_path = Path(args.model_path)
    else:
        # 3) Finally, download from HF as before
        model_path = Path(
            hf_hub_download(
                repo_id="MS-PINPOINT/mindglide",
                filename="_20240404_conjurer_trained_dice_7733.pt",
            )
        )

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

    if len(inp_files) == 0:
        print('Found 0 new images to segment. Exiting.')
        exit()

    # ===============================================
    # Prepare the datasets.
    # ===============================================

    # convert for MONAI dataset class formatting
    data = [{'image': f, 'output': o} for f,o in zip(inp_files, out_files) ]

    # Create MONAI dataset and dataloader
    # The transforms handle preprocessing like resizing and intensity normalization
    dataset = Dataset(data=data, transform=get_transforms(no_reorient=args.no_reorient))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Found {len(data)} images to process.")

    # ===============================================
    # Run the inference script
    # ===============================================

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Segmenting Images"):
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

            except Exception as e:
                print(f"⚠️ Error processing scan: {batch['image_meta_dict']['filename_or_obj'][0]}")
                print(f"Reason: {e}")
                continue

    print("\nInference complete. Segmentations saved to:", args.o)
