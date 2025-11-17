import os
import argparse
import warnings
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
    parser = argparse.ArgumentParser(description="MindGlide Brain Segmentation Inference")

    parser.add_argument('-i', type=str, required=True,
                        help='Path to a NIfTI file or a directory containing NIfTI images.')

    parser.add_argument('-o', type=str, required=True,
                        help='Path to the output NIfTI file or directory for saving segmentation masks.')

    parser.add_argument('--sw_batch_size', type=int, default=4,
                        help='Batch size for the sliding window inferer.')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='Ignore scans that have already been segmented')

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
    from mindglide.transforms import get_transforms, recovery_prediction
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

    # Download the weights from HF
    model_path = hf_hub_download(
        repo_id='MS-PINPOINT/mindglide', 
        filename='_20240404_conjurer_trained_dice_7733.pt'
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
    dataset = Dataset(data=data, transform=get_transforms())
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
                    
                    pred                = as_discrete(predictions[idx])
                    affine              = batch["image_meta_dict"]["affine"][idx].numpy()
                    resample_flag       = batch["resample_flag"][idx].item()
                    anisotrophy_flag    = batch["anisotrophy_flag"][idx].item()
                    crop_shape          = batch["crop_shape"][idx].tolist()
                    original_shape      = batch["original_shape"][idx].tolist()
                    bbox                = batch["bbox"][idx].tolist()

                    if resample_flag:
                        pred = recovery_prediction(pred, [num_classes, *crop_shape], anisotrophy_flag)

                    pred = np.argmax(pred, axis=0)

                    # Pad the cropped prediction back to the original image size
                    pred_padded = np.zeros(original_shape, dtype=pred.dtype)
                    (h_start, w_start, d_start), (h_end, w_end, d_end) = bbox
                    pred_padded[h_start:h_end, w_start:w_end, d_start:d_end] = pred

                    # Save the final segmentation map
                    nifti_img = nib.Nifti1Image(pred_padded.astype(np.uint8), affine)
                    nib.save(nifti_img, opaths[idx])

            except Exception as e:
                print(f"⚠️ Error processing scan: {batch['image_meta_dict']['filename_or_obj'][0]}")
                print(f"Reason: {e}")
                continue

    print("\nInference complete. Segmentations saved to:", args.o)
