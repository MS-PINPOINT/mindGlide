#! python
from scipy.stats import mode
import argparse
import json
import subprocess
import os
import ipdb
import shutil
import nibabel as nb
import numpy as np
from ensemble_utils import label_probabilities, \
    save_probabilities_nifti, vanilla_majority_vote, \
    generate_random_string


import pandas as pd


def calculate_volumes(seg_file_path):
    seg_img = nb.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_volume = np.prod(seg_img.header.get_zooms())

    unique_labels, counts = np.unique(seg_data, return_counts=True)
    volumes = {round(label): voxel_volume * count for label,
               count in zip(unique_labels, counts)}

    return volumes


def main(args):
    print('/mnt folder content: ', os.listdir('/mnt'))
    #dynamic_unet_folder = "/mounts/auto/arman7/workflows/monai/tutorials/modules/dynunet_pipeline/"
    dynamic_unet_folder = "/opt/monai-tutorials/modules/dynunet_pipeline/"
    container_shared_folder_with_host = '/mnt'
    print(f"model_file_paths: {args.model_file_paths}")
    # see task_prams.py in the dynunet pipeline for more info
    task_id = "12"  # 12 is the task id for under MONAI's dynunet pipeline.
    working_dir = container_shared_folder_with_host + "/tmpMINDGLIDE" \
        + generate_random_string(10)
    os.mkdir(working_dir)
    dynamic_unet_inference_path = f'{dynamic_unet_folder}/inference.py'
    if not os.path.isfile(dynamic_unet_inference_path):
        raise Exception("inference.py does not exist: ",
                        dynamic_unet_inference_path)
    command = f"python  {dynamic_unet_inference_path} -fold 0 \
-expr_name _mindglide -task_id {task_id} -tta_val False \
--root_dir {working_dir} \
--datalist_path {working_dir}"
    # output folder is created in the root dir with the name of _mindglide and the task id
    output_folder = working_dir + "/_mindglide" + task_id
    model_paths = args.model_file_paths
    # check if file paths exist
    for item in model_paths + [args.scan_path]:
        if not os.path.isfile(item):
            raise Exception("model file path does not exist: ", item)

    image_to_segment = args.scan_path
    if not '/mnt' in image_to_segment:
        image_to_segment = '/mnt/' + image_to_segment

    print("model_paths: ", model_paths)
    print("scan to segment: ", image_to_segment)

    if len(model_paths) > 1:
        ensemble_inference = True
    else:
        ensemble_inference = False
    # dataset json
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
                    'numTraining': 0,
                    'test': [{'image': image_to_segment}]
                    }
    with open(working_dir + "/dataset_task12.json", 'w') as outfile:
        json.dump(dataset_json, outfile)
    if not os.path.isfile(working_dir + "/dataset_task12.json"):
        raise Exception("dataset json file does not exist: ",
                        working_dir + "/dataset_task12.json")
    # run inference
    parent_output_folder = "runs_" + task_id + "_fold0__mindglide"
    output_folder = parent_output_folder + "/Task" + task_id + "_brain"
    output_file = os.path.join(
        output_folder,
        os.path.basename(image_to_segment)
    )
    if not ensemble_inference:
        model_path = model_paths[0]
        command = command + " --checkpoint " + model_path
        print(command)
        # os.system(command)
        # Run the script and capture its output
        result = subprocess.run(command,
                                shell=True,
                                capture_output=True, text=True)
        # Print the output of the script
        print("Output:", result.stdout)
        # Print the error (if any) of the script
        if result.stderr and len(result.stderr) > 0:
            print("Error:", result.stderr)
        # check if output file exists
        seg_file = container_shared_folder_with_host + '/' + \
            os.path.basename(image_to_segment).replace(
                '.nii.gz', '') + '-seg0.nii.gz'

        if not os.path.isfile(output_file):
            raise Exception("output file does not exist: ", output_file)
        else:
            shutil.move(output_file, seg_file)
    elif ensemble_inference:
        all_labels = []
        print("ensemble inference with ", len(model_paths), " models")
        for i, model_path in enumerate(model_paths):
            command = command + " --checkpoint " + model_path
            print(command)
            # os.system(command)
            # Run the script and capture its output
            result = subprocess.run(command,
                                    shell=True,
                                    capture_output=True, text=True)
            # Print the output of the script
            print("Output:", result.stdout)
            # Print the error (if any) of the script
            if result.stderr and result.returncode != 0:
                print("Error:", result.stderr)
            # check if output file exists
            if not os.path.isfile(output_file):
                raise Exception("output file does not exist: ", output_file)
            else:
                seg_file = container_shared_folder_with_host + '/' +\
                    os.path.basename(image_to_segment).replace('.nii.gz', '') + \
                    '-seg' + str(i) + '.nii.gz'
                shutil.move(output_file, seg_file)
                all_labels.append(nb.load(seg_file).get_fdata())
        #  majority voting
        #result = majority_vote(all_labels)
        result = vanilla_majority_vote(all_labels)
        seg_file = container_shared_folder_with_host + '/' +\
            os.path.basename(image_to_segment).replace('.nii.gz', '') + \
            '-segMajorityVoted.nii.gz'
        # save with nibabel
        img = nb.Nifti1Image(result,
                             affine=nb.load(image_to_segment).affine,
                             header=nb.load(image_to_segment).header)
        img.to_filename(seg_file)
        # probabilities
        label_prob_file = container_shared_folder_with_host + '/' +\
            os.path.basename(image_to_segment).replace('.nii.gz', '') + \
            '-labelProb.nii.gz'

        probabilities, unique_labels = label_probabilities(all_labels)
        save_probabilities_nifti(
            probabilities, image_to_segment, label_prob_file)
    # clean up
    print("cleaning up" + working_dir)
    # ipdb.set_trace()
    shutil.rmtree(working_dir)
    shutil.rmtree(parent_output_folder)

    # Save labels as a CSV file
    if os.path.isfile(seg_file):
        volumes = calculate_volumes(seg_file)
        print("Volumes: ", volumes)
    else:
        raise Exception("output file does not exist: ", seg_file)
    # Save labels as a CSV file
    labels_dict = dataset_json['labels']
    labels_df = pd.DataFrame(list(labels_dict.items()), columns=[
                             'Label_ID', 'Region_Name'])
    labels_df['Label_ID'] = labels_df['Label_ID'].astype(int)
    labels_df['Volume'] = labels_df['Label_ID'].map(volumes)
    csv_output_path = os.path.join(
        container_shared_folder_with_host, 'labels.csv')
    labels_df.to_csv(csv_output_path, index=False)


class CustomHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "Usage: "
        return super(CustomHelpFormatter, self).add_usage(usage, actions, groups, prefix)


if __name__ == "__main__":
    # Add your arguments
    parser = argparse.ArgumentParser(
        description='Process multiple file paths and another argument.',
        formatter_class=CustomHelpFormatter)
    parser.add_argument('--model_file_paths', metavar='FILE',
                        nargs='+', help='A list of file paths to dynamic u-net model. \
                            to pass multiple file paths, use space as a separator. if \
                            more than one file path is passed, the script will \
                            run inference for each model and average the results \
                            with ensemble learning.')
    parser.add_argument('--scan_path', type=str,
                        help='Nifti file path of the scan to segment.')
    parser.epilog = "example run: python run_inference.py \
                        --model_file_paths /path/to/model2.pt /path/to/model2.pt \
                        --scan_path /path/to/scan.nii.gz"
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
