import argparse
import os

import nibabel as nb
import numpy as np
import pandas as pd

from .consts import PROPERTIES


def calculate_volumes(seg_file_path):
    """Return {label_id: volume_mm3} for every label present in the image."""
    seg_img = nb.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_volume = np.prod(seg_img.header.get_zooms())

    unique_labels, counts = np.unique(seg_data, return_counts=True)
    volumes = {round(label): voxel_volume * count for label,
               count in zip(unique_labels, counts)}

    return volumes


def volumes_dataframe(seg_file_path):
    """Return a DataFrame with one row per MindGlide label and its volume in mm3."""
    volumes = calculate_volumes(seg_file_path)
    labels_dict = PROPERTIES["labels"]
    labels_df = pd.DataFrame(list(labels_dict.items()), columns=[
                             'Label_ID', 'Region_Name'])
    labels_df['Label_ID'] = labels_df['Label_ID'].astype(int)
    # Labels absent from the segmentation have zero volume (not NaN).
    labels_df['Volume_mm3'] = labels_df['Label_ID'].map(volumes).fillna(0.0)
    return labels_df


def main():
    parser = argparse.ArgumentParser(
        prog="mindglide-volumes",
        description="Calculate per-region volumes (mm3) from a MindGlide segmentation."
    )
    parser.add_argument(
        "label_file",
        help="Path to a NIfTI label image (e.g. mindglide segmentation output)",
    )
    parser.add_argument(
        "--out-csv",
        default="labels.csv",
        help="Output CSV path (default: labels.csv in current directory)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.label_file):
        parser.error(f"label file not found: {args.label_file}")

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    labels_df = volumes_dataframe(args.label_file)
    labels_df.to_csv(out_csv, index=False)
    print(f"Wrote volumes for {len(labels_df)} regions to: {out_csv}")


if __name__ == "__main__":
    main()
