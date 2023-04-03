import argparse
import os
from ensemble_utils import generate_random_string

container_shared_folder_with_host = "/mnt/"


def main(model_weight,
         image_list, label_list):
    # Your code to load the model with the weights and process the dataset
    print(f"Model weights: {model_weight}")
    train_fold0 = []
    for img, lbl in zip(image_list, label_list):
        train_fold0.append({'image': img,
                            'label': lbl})

    working_dir = container_shared_folder_with_host + \
        "/tmpMINDGLIDE" + \
        generate_random_string(10)

    model_paths = args.model_file_paths

    dynamic_unet_folder = "/opt/monai-tutorials/modules/dynunet_pipeline/"
    lr = 0.0001
    dataset_json = {'name': 'MindGlide',
                    'description': 'segments any modality into CGM, DGM and a few others',
                    'reference': 'MindGlide',
                    'licence': 'CC-BY-SA 4.0',
                    'release': '1.0 01/03/2023',
                    'tensorImageSize': '3D',
                    'modality': {'0': 'receives only one and any modality.'},
                    'labels': {'0': 'Background',
                               '1': 'CSF',
                               '2': 'Ventricle',
                               '3': 'DGM',
                               '4': 'Pons',
                               '5': 'Brain_stem',
                               '6': 'Cerebellum',
                               '7': 'Temporal_lobe',
                               '8': 'Ventricle',
                               '9': 'Lateral_Ventricle',
                               '10': 'Optic_Chiasm',
                               '11': 'Cerebellum',
                               '13': 'White_matter',
                               '12': 'Corpus_callosum',
                               '14': 'Frontal_lobe_GM',
                               '15': 'Limbic_cortex_GM',
                               '16': 'Parietal_lobe_GM',
                               '17': 'Occipital_lobe_GM',
                               '18': 'Lesion',
                               '19': 'Ventral_dc'},
                    'numTest': 1,
                    'numTraining': len(image_list),
                    'train_fold0': train_fold0
                    }

    command = dynamic_unet_folder + "/train.py \
         -train_num_workers 4 -interval 1 -num_samples 3  \
          --task_id 12 --root_dir ${working_dir} \
          -learning_rate ${lr} \
             -max_epochs 100 \
            -pos_sample_num 2 -expr_name _mindglide \
         -tta_val False --datalist_path \
          ${working_dir} --fold 0 \
            -checkpoint ${MOST_RECENT_WEIGHT_FILE}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load PyTorch model with specified weights and process dataset.")

    parser.add_argument(
        "--model_weight",
        type=str,
        required=True,
        help="Path to the PyTorch weights to be loaded into the state_dict",
    )

    parser.add_argument(
        "--image_list",
        type=str,
        nargs='+',
        required=True,
        help="List of image paths (strings)",
    )
    parser.add_argument(
        "--label_list",
        type=str,
        nargs='+',
        required=True,
        help="List of label paths (strings) corresponding to the images",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model_weight):
        print(f"Error: model_weights file not found at {args.model_weights}")
        exit(1)

    main(args.model_weight,
         args.image_list, args.label_list)
