# MindGlide

S<mark>e</mark>g<mark>m</mark>enting clinical <mark>M</mark>R<mark>I</mark> scans of <mark>m</mark>ultiple sclerosis patients of any quality and <mark>mo</mark>dality

MindGlide is a deep learning model for ultrafast segmentation of real-world, brain MRI scans of any modality and any quality. MindGlide does not require any extra preprocessing or postprocessing. It can be used for research applications, such as lesion segmentation, brain structure quantification, and brain atrophy analysis.

<p align="center">
<img src="assets/t2.png" alt="MindGlide logo" width="500" height="300">
</p>

MindGlide is an open source PyTorch model built with the MONAI framework
on more than 23,000 scans from multiple sclerosis patients. MindGlide
quantifies the volume of brain structures and lesions using by ultrafast
segmentation on GPU hardware.

## Quick start

## Requirements

You will need to install the following software:

- Git
- Docker
- Git LFS (Large File Storage)
- AppTainer (optional)

### Installation

MindGlide can be run without installation using container technology. The
following command will pull the latest version of the container from
GitHub's Large File Storage (LFS) and run it interactively:

`git clone https://github.com/MS-PINPOINT/mindGlide.git`

PyTorch trained models are stored in the `models` directory. The
following command will download the latest version of the models:

`git lfs pull`

You can use Docker or Apptainer to run the container. To start testing
MindGlide, run the following command:

```
cd mindGlide
docker run -it --rm -v $(pwd):/mindGlide -w /mindGlide armaneshaghi/ms-pinpoint/mind-glide:latest {name_of_nifti_file}
```

You need to replace `{name_of_nifti_file}` with the name of the NIfTI file. For example, if you want to run the test file `test.nii.gz`, you can run the following command:

```
docker run -it --rm -v $(pwd):/mindGlide -w /mindGlide armaneshaghi/ms-pinpoint/mind-glide:latest test.nii.gz
```

`test.nii.gz` will be a brain MRI file.

Docker container is used for testing the model. If you want to use the
container on High Performance Computing (HPC) clusters, you can use
Apptainer (formerly known as Singularity). See [Apptainer](https://apptainer.org/) documentations for details.

<p align="center">
<img src="assets/mindGlide_logo.png" alt="MindGlide logo" width="300" height="300">
</p>

### Usage

#### With Docker

```
docker run --gpus all \
--ipc=host --ulimit memlock=-1 -it \
-v $PWD:/mnt \
armaneshaghi/mind-glide:latest test/flair.nii.gz
```

#### With Apptainer (Singularity)

First you need to use another Docker container to build the Apptainer or Singularity image:

`docker run -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/test:/output --privileged -it --rm  quay.io/singularity/docker2singularity armaneshaghi/mind-glide:latest`

This will create a Singularity image in the `/tmp/test` directory.

Then you can run the Singularity image, we assume it is called `mind-glide_latest.sif`
which had resulted from the above step:

```
#running from the same directory as the image
singularity run --nv \
--bind $PWD:/mnt \
/path/to/mind-glide_latest.sif flair.nii.gz
```

### Fine tuning

MindGlide models can be fine-tuned on your own data. Fine tuning is useful when you are not getting the desired results on your data. You can fine tune the model on your data by starting from the scripts in the
`scripts` directory.

_Parameters for fine tuning_: The shipped models with the repo are trained a learning rate (lR)
of 0.01 . If you want to fine tune the model on your data, you can start with a lower learning rate, such as 0.001. Using the same learning rate as the shipped models may result in **catastrophic forgetting**.

### Modifications

If you want to run the Docker container using your own trained model or use
a custom script, please make sure you overwrite teh `entrypoint`. For example,
to run bash and get into the container, you can run the following command:

`docker run -it --entrypoint bash armaneshaghi/mind-glide:latest`

### Shared models

Several trained models are shared in the `models` directory.
They are trained on the datasets explained in the paper (link to be added upon publication).

Tailored models for extermely low quality data, that is MRI data with disproportionate small FOV, and large gaps between
slices are shared separetly. The following table provides
more information about the models:

| Model name and path in the models directory | Description | Dataset |
|-|-|-|
| /opt/mindGlide/models/_20240404_conjurer_trained_dice_7733.pt | trained on dataset 1 (This model is 10 times larger than the previous models) | IPMSA |

### 📬 Citation

If you use MindGlide please cite this paper:

```bibtex
@article {Goebl2024.03.29.24305083,
	author = {Philipp Goebl and Jed Wingrove and Omar Abdelmannan and Barbara Brito Vega and Jonathan Stutters and Silvia Da Graca Ramos and Owain Kenway and Thomas Rosoor and Evangeline Wassmer and Jeremy Chataway and Douglas Arnold and Louis Collins and Cheryl Hemmingway and Sridar Narayanan and Declan Chard and Juan Eugenio Iglesias and Frederik Barkhof and Yael Hacohen and Alan Thompson and Daniel Alexander and Olga Ciccarelli and Arman Eshaghi},
	title = {Repurposing Clinical MRI Archives for Multiple Sclerosis Research with a Flexible, Single-Modality Approach: New Insights from Old Scans},
	elocation-id = {2024.03.29.24305083},
	year = {2024},
	doi = {10.1101/2024.03.29.24305083},
	URL = {https://www.medrxiv.org/content/early/2024/03/30/2024.03.29.24305083},
	eprint = {https://www.medrxiv.org/content/early/2024/03/30/2024.03.29.24305083.full.pdf},
	journal = {medRxiv}
} 
```

### Acknowledgements

This study/project is funded by the UK National Institute for Health and Social Care (NIHR) Advanced Fellowship to Arman Eshaghi (Award ID: NIHR302495). The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.

<p align="left">
<img src="assets/nihr_logo.png" alt="NIHR logo" >
</p>

Credits: Image is created by OpenAI Dall-E2.
