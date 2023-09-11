import argparse
import json
import os
from ensemble_utils import generate_random_string
import shutil

import ipdb

container_shared_folder_with_host = "/mnt/"

os.environ["MKL_THREADING_LAYER"] = "GNU"


def main(model_weight,
         training_image_list, training_label_list,
         validation_image_list,
         validation_label_list):
    # Your code to load the model with the weights and process the dataset
    print(f"Model weights: {model_weight}")
    print(f"Training Image list: {training_image_list}")
    print(f"Training Label list: {training_label_list}")
    train_fold0 = []
    validation_fold0 = []
    for img, lbl in zip(training_image_list, training_label_list):
        train_fold0.append({'image': img,
                            'label': lbl})
    for img, lbl in zip(validation_image_list, validation_label_list):
        validation_fold0.append({'image': img,
                                 'label': lbl})
    working_dir = container_shared_folder_with_host + \
        "/tmpMINDGLIDE" + \
        generate_random_string(10)
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
        print("Created working directory: ", working_dir)
    dynamic_unet_folder = "/opt/monai-tutorials/modules/dynunet_pipeline/"
    lr = 0.00001
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
                    'numTraining': len(training_image_list),
                    "training": train_fold0,
                    'train_fold0': train_fold0,
                    'validation_fold0': validation_fold0
                    }
    with open(working_dir + "/dataset_task12.json", 'w') as f:
        json.dump(dataset_json, f)
    epochs = 10
    command = ('python ' + dynamic_unet_folder + "/train.py "
               f"-train_num_workers 4 -interval 1 -num_samples 3 "
               f" --task_id 12 --root_dir {working_dir} "
               f"-learning_rate {lr} "
               f" -max_epochs {epochs}"
               f" -pos_sample_num 2 -expr_name _mindglide "
               f" -tta_val False --datalist_path "
               f" {working_dir} --fold 0 "
               f"-checkpoint {model_weight}")
    print(command)
    os.system(command)
    shutil.rmtree(working_dir)
    #
    # ipdb.set_trace()
    output_dir = "/mnt/runs_12_fold0__mindglide"
    if not os.path.exists(output_dir):
        # throw error
        print("Error: output directory not found at ", output_dir)
    else:
        date = output_dir.split("_")[-1]
        shutil.move(output_dir, 'fine_tuning_output_' + date)


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
        "--training_image_list",
        type=str,
        nargs='+',
        required=True,
        help="List of image paths (strings)",
    )
    parser.add_argument(
        "--training_label_list",
        type=str,
        nargs='+',
        required=True,
        help="List of label paths (strings) corresponding to the images",
    )

    parser.add_argument(
        "--validation_image_list",
        type=str,
        nargs='+',
        required=True,
        help="List of image paths (strings)",
    )
    parser.add_argument(
        "--validation_label_list",
        type=str,
        nargs='+',
        required=True,
        help="List of label paths (strings) corresponding to the images",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model_weight):
        print(f"Error: model_weights file not found at {args.model_weight}")
        exit(1)

    main(args.model_weight,
         args.training_image_list, args.training_label_list,
         args.validation_image_list,
         args.validation_label_list)
