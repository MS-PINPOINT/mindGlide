# MindGlide

<p align="center">
<img src="assets/mindGlide_logo.png" alt="MindGlide logo" width="300" height="300">
</p>

MindGlide is an open source PyTorch model built with the MONAI framework
on more than 23,000 scans from multiple sclerosis patients. MindGlide
quantifies the volume of brain structures and lesions using by ultrafast
segmentation on GPU hardware.

## Quick Start

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
Apptainer (formerly known as Singularity). See Apptainer documentations for details.

### Acknowledgements

This study/project is funded by the UK National Institute for Health and Social Care (NIHR) Advanced Fellowship to Arman Eshaghi (). The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.

<p align="left">
<img src="assets/nihr_logo.png" alt="NIHR logo" >
</p>

Credits: Image is created by OpenAI Dall-E2.

```

```
