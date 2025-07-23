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
- Huggingface
- Docker
- AppTainer (optional)

### Installation

MindGlide can be run without installation using container technology. The
following command will pull the latest version of the container from
GitHub's Large File Storage (LFS) and run it interactively:

`git clone https://github.com/MS-PINPOINT/mindGlide.git`

PyTorch trained models are stored on Huggingface:
`https://huggingface.co/MS-PINPOINT/mindglide/tree/main`

You can use Docker or Apptainer to run the container. 

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
mspinpoint/mindglide:may2024 test/flair.nii.gz
```

#### With Apptainer (Singularity)

First you need to use another Docker container to build the Apptainer or Singularity image:

```
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/test:/output --privileged -it --rm  quay.io/singularity/docker2singularity mspinpoint/mindglide:may2024
```

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

`docker run -it --entrypoint bash mspinpoint/mindglide:may2024`

### Shared models

Trained models are shared in the `models` directory.
They are trained on the datasets explained in the paper [**Nature Communications (2025)**](https://www.nature.com/articles/s41467-025-58274-8#citeas).

Tailored models for extermely low quality data, that is MRI data with disproportionate small FOV, and large gaps between
slices are shared separetly. The following table provides
more information about the models:

| Model name and path in the models directory | Description | Dataset |
|-|-|-|
| /opt/mindGlide/models/_20240404_conjurer_trained_dice_7733.pt | trained on dataset 1 (This model is 10 times larger than the previous models) | IPMSA |

### 📬 Citation

If you use MindGlide please cite this paper:

```bibtex
@article{Goebl2025,
    author = {Goebl, Philipp and Wingrove, Jed and Abdelmannan, Omar and {Brito Vega}, Barbara and Stutters, Jonathan and Ramos, {Silvia Da Graca} and Kenway, Owain and Rossor, Thomas and Wassmer, Evangeline and Arnold, Douglas L. and Collins, Louis and Hemingway, Cheryl and Narayanan, Sridar and Chataway, Jeremy and Chard, Declan and Iglesias, {Juan Eugenio} and Barkhof, Frederik and Parker, Geoffrey J. M. and Oxtoby, Neil P. and Hacohen, Yael and Thompson, Alan and Alexander, Daniel C. and Ciccarelli, Olga and Eshaghi, Arman},
    title = {Enabling new insights from old scans by repurposing clinical {MRI} archives for multiple sclerosis research},
    journal = {Nature Communications},
    volume = {16},
    number = {1},
    pages = {3149},
    year = {2025},
    month = apr,
    doi = {10.1038/s41467-025-58274-8},
    pmid = {40195318},
    pmcid = {PMC11976987}
}
```

### Acknowledgements

This study/project is funded by the UK National Institute for Health and Social Care (NIHR) Advanced Fellowship to Arman Eshaghi (Award ID: NIHR302495). The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.

<p align="left">
<img src="assets/nihr_logo.png" alt="NIHR logo" >
</p>

Credits: Image is created by OpenAI Dall-E2.