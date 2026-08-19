# MindGlide

The Universal Brain MRI Segmentation Tool for Multiple Sclerosis.

[![PyPI](https://img.shields.io/pypi/v/mindglide?style=flat-square&color=blue)](https://pypi.org/project/mindglide/)
[![CI](https://img.shields.io/github/actions/workflow/status/MS-PINPOINT/mindGlide/ci.yml?style=flat-square&logo=github)](https://github.com/MS-PINPOINT/mindGlide/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://github.com/MS-PINPOINT/mindGlide/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--58274--8-blue?style=flat-square)](https://doi.org/10.1038/s41467-025-58274-8)

Built with PyTorch + MONAI | Trained on >23,000 real-world scans | Published in [Nature Communications](https://doi.org/10.1038/s41467-025-58274-8) | Project by [MS-PINPOINT](https://www.ms-pinpoint.com)

<img src="https://raw.githubusercontent.com/MS-PINPOINT/mindGlide/main/assets/any_sequence_any_quality.png" alt="MindGlide segmentations of five very different inputs" width="100%">
*One model, no preprocessing, no retuning — from a 64 mT portable scanner to 7 T, research-grade 3D to thick-slice clinical 2D. Openly licensed public scans; unedited MindGlide output.*

## What does it do? (Supported Inputs)
- **Any single MRI sequence:** T1, T2, FLAIR, PD, post-contrast; one image per scan (no multi-channel input needed).
- **Any quality:** Designed for real-world clinical archives (2D thick-slice, anisotropic voxels, older scans) and research-grade 3D images. Resampling and reorientation happen internally; output always lands back on the input grid.
- **Zero Preprocessing:** Feed it raw NIfTI files. No skull-stripping, no bias correction, no registration, no reorienting needed.

> [!IMPORTANT]
> **Intended use**: Research only. MindGlide is not a medical device and must not be used for clinical decision-making.

## Quickstart

**1. Install**
```bash
pip install mindglide
```

**2. Run**
```bash
mindglide -i scan.nii.gz -o scan_seg.nii.gz
```
*The model (~123 MB) downloads and caches automatically on the first run.*

**Process an entire folder:**
```bash
mindglide -i scans/ -o segs/
```

**Try with public MNI152 template:**
```bash
curl -O https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz
mindglide -i tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz -o mni_seg.nii.gz
```

<img src="https://raw.githubusercontent.com/MS-PINPOINT/mindGlide/main/assets/mni_overlay.png" alt="MindGlide segmentation of the MNI152 template: input scan vs segmented output in three views" width="640">

## Key Features & Python API

MindGlide provides a seamless Python API that guarantees byte-identical outputs to the CLI.

```python
from mindglide import segment, volumes_dataframe
import pandas as pd

# Segment a single scan
seg_path = segment("scan.nii.gz")

# Get region volumes in mm³
df = volumes_dataframe(seg_path)

# Process a folder and extract cohort statistics
segment("scans_dir/", "segs_dir/")
# Equivalent to: mindglide-volumes segs_dir/ --out-csv cohort.csv
cohort_df = pd.read_csv("cohort.csv")
```

## Options & Configuration

| Option / Env Var | Meaning |
|---|---|
| `--device {auto,cpu,cuda,mps}` | Compute device (default: auto). |
| `--sw-batch-size N` | Sliding-window batch size (default 4). Lower it if the GPU runs out of memory. |
| `--model-path FILE` / `MODEL_PATH` | Use a local `.pt` checkpoint instead of the automatic download (offline use). |
| `HF_HOME` | Model cache location for auto-downloads (default: `~/.cache/huggingface`). |
| `--resume` | Skip scans whose segmentation already exists at the output location. |
| `--no-klc` | Keep all connected components (skip largest-component cleanup). |
| `--no-reorient` | Skip internal RAS re-orientation. Output always matches the input scan's grid. |

## Agent-Ready & Integration Guide
For LLM agents or automated pipelines integrating MindGlide:
- Ensure the environment variable `HF_HOME` is set if cache persistence is needed across ephemeral containers.
- Use the Python API (`from mindglide import segment`) and handle `mindglide.UsageError` for predictable failure states.
- For high-throughput environments, set `MODEL_PATH` to a pre-downloaded weights file to bypass network calls.

## Troubleshooting

> [!TIP]
> **GPU out of memory?** Try setting `--sw-batch-size 1`, or fallback to `--device cpu`.

**"Warning: not using the GPU"**
Default `pip` PyTorch wheels no longer include kernels for older GPUs (e.g. Pascal cards). Install a compatible PyTorch first:
```bash
pip install "torch==2.6.0+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install mindglide
```

**Apple Silicon**
`auto` uses MPS when available. If an operation is unsupported, run with `--device cpu` or set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Deployment: Docker & Apptainer

Prebuilt images (~8 GB with CUDA runtime) are available for offline or containerized workloads:

**Docker:**
```bash
docker pull ghcr.io/ms-pinpoint/mindglide:latest
docker run --gpus all --ipc=host --user $(id -u):$(id -g) -v /data:/data \
  ghcr.io/ms-pinpoint/mindglide:latest -i /data/scan.nii.gz -o /data/scan_seg.nii.gz
```

**Apptainer (HPC):**
```bash
apptainer pull mindglide.sif docker://ghcr.io/ms-pinpoint/mindglide:latest
apptainer run --nv -B /data:/data mindglide.sif -i /data/scan.nii.gz -o /data/scan_seg.nii.gz
```

## Model Weights & Output Labels

The model outputs 19 anatomical regions + background. See `mindglide --labels` for the full index. Colormaps for FSLeyes and FreeSurfer are available in the `labels/` directory.

The default checkpoint is pinned for reproducibility and downloads from [Hugging Face](https://huggingface.co/MS-PINPOINT/mindglide).

## Development & Tests

```bash
git clone https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
pip install -e ".[test]"

pytest                          # fast unit tests
MINDGLIDE_RUN_SLOW=1 pytest -v  # + end-to-end on public MNI scan
```
See `CONTRIBUTING.md` for guidelines.

## Citation

If you use MindGlide, please cite:

> Goebl P, Wingrove J, Abdelmannan O, *et al.* Enabling new insights from old scans by repurposing clinical MRI archives for multiple sclerosis research. *Nature Communications*. 2025;16(1):3149. doi:10.1038/s41467-025-58274-8

<details>
<summary>BibTeX</summary>

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

</details>

## Acknowledgements

MindGlide is a flagship project of the [MS-PINPOINT](https://www.ms-pinpoint.com) group.

This study and the initial development of this project were funded by the UK National Institute for Health and Care Research (NIHR) Advanced Fellowship to Arman Eshaghi (Award ID: NIHR302495). The team is currently based at the **UCL Hawkes Institute** and the UCL Queen Square Multiple Sclerosis Centre.

In **July 2027**, the core team and this research will transition to the **DreaMS Lab** at **King's College London** (Institute of Psychiatry, Psychology & Neuroscience), backed by an 8-year project funded by **Wellcome**.

<img src="https://raw.githubusercontent.com/MS-PINPOINT/mindGlide/main/assets/nihr_logo.png" alt="NIHR logo" width="200">
