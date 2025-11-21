import argparse
import nibabel as nb
import numpy as np 

def calculate_volumes(seg_file_path):
    seg_img = nb.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_volume = np.prod(seg_img.header.get_zooms())

    unique_labels, counts = np.unique(seg_data, return_counts=True)
    volumes = {round(label): voxel_volume * count for label,
               count in zip(unique_labels, counts)}

    return volumes
def main():
    parser = argparse.ArgumentParser(
        description="Calculate volumes from a label image"
    )
    parser.add_argument(
        "label_file",
        help="Path to a NIfTI label image (e.g. segmentation output)",
    )

    args = parser.parse_args() 

    label_file = args.label_file
    volumes = calculate_volumes(label_file)
    
    labels_dict = dataset_json['labels']
    labels_df = pd.DataFrame(list(labels_dict.items()), columns=[
                             'Label_ID', 'Region_Name'])
    labels_df['Label_ID'] = labels_df['Label_ID'].astype(int)
    labels_df['Volume'] = labels_df['Label_ID'].map(volumes)
    csv_output_path = os.path.join(
        container_shared_folder_with_host, 'labels.csv')
    labels_df.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    main()
