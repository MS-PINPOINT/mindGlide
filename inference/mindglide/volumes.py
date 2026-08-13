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
    # Only the spatial zooms: 4D images (e.g. a trailing singleton volume
    # dimension) report a temporal zoom too, which must not scale volumes.
    voxel_volume = float(np.prod(seg_img.header.get_zooms()[:3]))

    unique_labels, counts = np.unique(seg_data, return_counts=True)
    finite = np.isfinite(unique_labels)
    if not finite.all():
        print("Warning: ignoring non-finite voxel values (NaN/inf) in the label image.")
        unique_labels, counts = unique_labels[finite], counts[finite]
    volumes = {round(label): voxel_volume * count for label,
               count in zip(unique_labels, counts)}

    return volumes


def looks_like_segmentation(seg_file_path, max_label=19, max_distinct=64):
    """
    Heuristic check that a NIfTI file is a label map rather than a raw scan:
    integer-valued voxels, labels within the MindGlide range, few distinct values.
    """
    data = nb.load(seg_file_path).get_fdata()
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    distinct = np.unique(finite)
    return (
        len(distinct) <= max_distinct
        and np.allclose(distinct, np.round(distinct))
        and distinct.max() <= max_label
        and distinct.min() >= 0
    )


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
    try:
        nb.load(args.label_file)
    except Exception as e:
        parser.error(f"not a readable NIfTI file: {args.label_file} ({e})")

    if not looks_like_segmentation(args.label_file):
        print(f"Warning: {args.label_file} does not look like a MindGlide segmentation "
              "(non-integer values or labels outside 0-19) — did you pass the raw scan "
              "instead of the *_seg output? Writing volumes anyway.")

    out_csv = os.path.abspath(args.out_csv)
    if os.path.isdir(out_csv):
        out_csv = os.path.join(out_csv, "labels.csv")
        print(f"Note: --out-csv is a directory; writing {out_csv}")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    labels_df = volumes_dataframe(args.label_file)
    labels_df.to_csv(out_csv, index=False)
    print(f"Wrote volumes for {len(labels_df)} regions to: {out_csv}")


if __name__ == "__main__":
    main()
