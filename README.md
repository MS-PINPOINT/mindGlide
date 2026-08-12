<div align="center">

# MindGlide

<p>
<strong>Ultrafast segmentation of real‑world brain MRI for multiple‑sclerosis patients — any modality, any quality.</strong><br>Built with PyTorch + MONAI and trained on >23 000 scans.<br>
</p>

<p align="center">
  <img src="assets/t2.png" alt="MindGlide banner" width="500" height="300">
</p>
</div>


## Quickstart

Requirements: Python ≥ 3.9 and `pip`. A GPU is **not** required — MindGlide
runs in seconds on a GPU and typically in a few minutes per scan on a CPU.

```bash
# 1) install (in a virtual environment if you prefer)
pip install git+https://github.com/MS-PINPOINT/mindGlide.git

# 2) check the command is available
mindglide --help

# 3) segment a scan
mindglide -i /path/to/scan.nii.gz -o /path/to/scan_seg.nii.gz
```

The first run downloads the trained model (~123 MB) automatically from the
[Hugging Face Hub](https://huggingface.co/MS-PINPOINT/mindglide) and caches it,
so later runs work offline.

No scan at hand? Try it on the public MNI152 template:

```bash
curl -O https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz
mindglide -i tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz -o mni_seg.nii.gz
```

Open the result on top of the input in your favourite viewer (FSLeyes, ITK-SNAP,
freeview…) to inspect the 19 segmented regions.


## Usage

Segment a single scan:

```bash
mindglide -i scan.nii.gz -o scan_seg.nii.gz
```

Segment every NIfTI file in a directory (the model is loaded once; outputs are
named `<scan>_seg.nii.gz`; non-NIfTI files are ignored):

```bash
mindglide -i scans_dir/ -o segmentations_dir/
# add --resume to skip scans already segmented in the output directory
```

Useful options:

| Option | Meaning |
|---|---|
| `--device {auto,cpu,cuda,mps}` | Compute device. Default `auto` picks a working GPU if present, else CPU. |
| `--sw-batch-size N` | Sliding-window batch size (default 4). Lower it if you run out of GPU memory. |
| `--model-path FILE` | Use a local `.pt` checkpoint instead of the automatic download (useful offline). |
| `--resume` | Directory mode: skip scans already segmented. |
| `--no-klc` | Keep all connected components (skip largest-component cleanup). |
| `--no-reorient` | Skip internal RAS re-orientation. Output always matches the input scan's grid. |
| `--version` | Print the installed version. |

### Per-region volumes

Turn a segmentation into a CSV of region volumes (mm³):

```bash
mindglide-volumes scan_seg.nii.gz --out-csv scan_volumes.csv
```

### Label map

| Code | Structure Name                  | Code | Structure Name           |
|:----:|:--------------------------------|:----:|:--------------------------|
| 0    | Background                      | 10   | Optic_chiasm              |
| 1    | CSF                             | 11   | Cerebellar_vermis         |
| 2    | Ventricles_3_4_5                | 12   | Corpus_callosum           |
| 3    | DGM                             | 13   | White_matter              |
| 4    | Pons                            | 14   | Frontal_lobe_GM           |
| 5    | Brainstem                       | 15   | Limbic_cortex_GM          |
| 6    | Cerebellum                      | 16   | Parietal_lobe_GM          |
| 7    | Temporal_lobe                   | 17   | Occipital_lobe_GM         |
| 8    | Temporal_horn_lateral_ventricle | 18   | Lesion                    |
| 9    | Lateral_ventricle               | 19   | Ventral_diencephalon      |


## Troubleshooting

**"Warning: not using the GPU — … this PyTorch build cannot run on it"** —
the default `pip` PyTorch wheels no longer ship kernels for older GPU
architectures (e.g. Pascal cards: GTX 10xx, Quadro P series). MindGlide falls
back to CPU automatically. To use such a GPU, install a PyTorch build with an
older CUDA variant first, then install MindGlide:

```bash
pip install "torch==2.6.0+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/MS-PINPOINT/mindGlide.git
```

**GPU out of memory** — lower the sliding-window batch size
(`--sw-batch-size 1`) or run with `--device cpu`.

**Apple Silicon (M-series)** — `auto` uses MPS when available. If you hit an
unsupported-operation error, run with `--device cpu` or set
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Offline / air-gapped machines** — download
[`_20240404_conjurer_trained_dice_7733.pt`](https://huggingface.co/MS-PINPOINT/mindglide/tree/main)
once and pass it with `--model-path /path/to/model.pt` (or set the
`MODEL_PATH` environment variable).


## Run in Docker

Build the image once (bakes in the model weights, so the container runs
offline):

```bash
git clone https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
docker build -t mindglide .
```

Then segment scans in a mounted folder:

```bash
# with a GPU
docker run --gpus all --ipc=host -v /data:/data \
  mindglide -i /data/scan.nii.gz -o /data/scan_seg.nii.gz

# CPU only
docker run --ipc=host -v /data:/data \
  mindglide -i /data/scan.nii.gz -o /data/scan_seg.nii.gz
```

For Singularity/Apptainer on HPC, build from the same image:

```bash
apptainer build mindglide.sif docker-daemon://mindglide:latest
apptainer run --nv -B /data:/data mindglide.sif -i /data/scan.nii.gz -o /data/scan_seg.nii.gz
```


## Model weights

The primary checkpoint is downloaded automatically on first run and lives at
[Hugging Face: MS-PINPOINT/mindglide](https://huggingface.co/MS-PINPOINT/mindglide)
(`_20240404_conjurer_trained_dice_7733.pt`). Additional or legacy checkpoints
are archived in the same repository. The models were trained on the datasets
described in the paper
[**Nature Communications (2025)**](https://www.nature.com/articles/s41467-025-58274-8#citeas).

If you work from a source checkout, you can also fetch the weights as a git
submodule (requires [Git LFS](https://git-lfs.com)):

```bash
git clone --recurse-submodules https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
git lfs install            # first time only
git submodule foreach 'git lfs pull'
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This places `models/_20240404_conjurer_trained_dice_7733.pt` in the workspace.


## Development and tests

```bash
git clone https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
pip install -e ".[test]"

# fast unit tests (no model download, a few seconds)
pytest

# full end-to-end tests: download the model and segment a public MNI template
# on CPU — and on GPU when one is available
MINDGLIDE_RUN_SLOW=1 pytest -v
```


## Fine‑tuning

Use the scripts in `scripts/` as a template. Start with a low learning
rate (e.g. 1e‑3) to avoid catastrophic forgetting — shipped models were
trained with 1e‑2.


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


---

## Acknowledgements

This study/project is funded by the UK National Institute for Health and
Social Care (NIHR) Advanced Fellowship to Arman Eshaghi (Award ID:
NIHR302495). The views expressed are those of the author(s) and not
necessarily those of the NIHR or the Department of Health and Social
Care.

<p align="left">
  <img src="assets/nihr_logo.png" alt="NIHR logo" width="200">
</p>
