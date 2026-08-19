<div align="center">

# 🧠 MindGlide

### The Universal Brain MRI Segmentation Tool for Multiple Sclerosis
**Any Sequence. Any Quality. Zero Preprocessing.**

Built with **PyTorch + MONAI** | Trained on **>23,000** real-world scans | Published in **[Nature Communications](https://doi.org/10.1038/s41467-025-58274-8)** | Project by **[MS-PINPOINT](https://www.ms-pinpoint.com)**

[![PyPI](https://img.shields.io/pypi/v/mindglide?style=flat-square&color=blue)](https://pypi.org/project/mindglide/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MS-PINPOINT/mindGlide/blob/main/examples/mindglide_quickstart.ipynb)
[![CI](https://img.shields.io/github/actions/workflow/status/MS-PINPOINT/mindGlide/ci.yml?style=flat-square&logo=github)](https://github.com/MS-PINPOINT/mindGlide/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://github.com/MS-PINPOINT/mindGlide/blob/main/LICENSE)
[![Python ≥3.9](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg?style=flat-square&logo=python&logoColor=white)](https://github.com/MS-PINPOINT/mindGlide)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--58274--8-blue?style=flat-square)](https://doi.org/10.1038/s41467-025-58274-8)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-MS--PINPOINT%2Fmindglide-orange?style=flat-square)](https://huggingface.co/MS-PINPOINT/mindglide)

<br>

<img src="https://raw.githubusercontent.com/MS-PINPOINT/mindGlide/main/assets/any_sequence_any_quality.png" alt="MindGlide segmentations of five very different inputs" width="100%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

*One model, no preprocessing, no retuning — from a 64 mT portable scanner to 7 T, research-grade 3D to thick-slice clinical 2D. Openly licensed public scans; unedited MindGlide output.*
</div>

<br>

## ✨ Why MindGlide?
- 🚀 **Zero Preprocessing:** Feed it raw NIfTI files. No skull-stripping, no bias correction, no registration, no reorienting needed.
- 💻 **Dead Simple:** One command to segment a single scan or an entire folder.
- ⚡ **Lightning Fast:** Seconds per scan on a GPU, or just a few minutes on a CPU.
- 🏥 **Robust:** Handles clinical archives out-of-the-box, no matter how old or unusual the sequence.

---

## 🚀 Get Started in 3 Seconds

You are just one command away from segmented scans.

```bash
# 1. Install
pip install mindglide

# 2. Run
mindglide -i scan.nii.gz -o scan_seg.nii.gz
```

That's it! The model (~123 MB) downloads and caches automatically on the first run.

### 📂 Process an entire folder instantly
Got an archive of scans? MindGlide handles it gracefully:
```bash
mindglide -i scans/ -o segs/   # writes segs/<name>_seg.nii.gz for every scan
```

Prefer zero installs? **[Try it in your browser on Colab →](https://colab.research.google.com/github/MS-PINPOINT/mindGlide/blob/main/examples/mindglide_quickstart.ipynb)**

### 🧠 No scan at hand? Try the public MNI152 template:
```bash
curl -O https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz
mindglide -i tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz -o mni_seg.nii.gz
```

<div align="center">
<img src="https://raw.githubusercontent.com/MS-PINPOINT/mindGlide/main/assets/mni_overlay.png" alt="MindGlide segmentation of the MNI152 template: input scan vs segmented output in three views" width="640">

*The command above produces the segmented MNI152 template.*
</div>

---

## 🛠️ Seamless Python API

Integrate MindGlide directly into your workflow. It uses the same engine as the CLI, guarantees byte-identical outputs, and provides clean exceptions (`mindglide.UsageError`) instead of exit codes.

```python
from mindglide import segment, volumes_dataframe

# Segment a single scan
seg_path = segment("scan.nii.gz")            # writes scan_seg.nii.gz

# Instantly get region volumes in mm³
df = volumes_dataframe(seg_path)             

# Or process a whole folder
segment("scans_dir/", "segs_dir/")           
```

---

## 📊 From Scans to Statistics

Segment a cohort, then get **one CSV for the whole study**:

```bash
mindglide -i scans/ -o segs/ --resume        # resumable folder-mode segmentation
mindglide-volumes segs/ --out-csv cohort.csv # one long-format table for all scans
```

```python
import pandas as pd
df = pd.read_csv("cohort.csv")               # columns: Scan, Label_ID, Region_Name, Volume_mm3
lesions = df[df.Region_Name == "Lesion"]     # e.g. lesion volume per scan
```

## Options

| Option | Meaning |
|---|---|
| `--device {auto,cpu,cuda,mps}` | Compute device (default: auto — a working GPU if present, else CPU). |
| `--sw-batch-size N` | Sliding-window batch size (default 4). Lower it if the GPU runs out of memory. |
| `--model-path FILE` | Use a local `.pt` checkpoint instead of the automatic download (offline use). |
| `--resume` | Skip scans whose segmentation already exists at the output location. |
| `--no-klc` | Keep all connected components (skip largest-component cleanup). |
| `--no-reorient` | Skip internal RAS re-orientation. Output always matches the input scan's grid. |
| `--labels` | Print the label code / region name table and exit. |

## Output labels

19 regions + background (`mindglide --labels` prints this table):

| Code | Structure                       | Code | Structure                 |
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

**See named, colored regions in your viewer** — ready-made colormaps live in
[`labels/`](labels/):

```bash
fsleyes scan.nii.gz scan_seg.nii.gz -ot label -l labels/mindglide_fsleyes.lut
freeview -v scan.nii.gz scan_seg.nii.gz:colormap=lut:lut=labels/mindglide_freesurfer.txt
# ITK-SNAP: Segmentation > Label Editor > Actions > Import label descriptions
```

## What can I feed it?

- **Any single MRI sequence** — T1, T2, FLAIR, PD, post-contrast; one image per
  scan (no multi-channel input needed).
- **Any quality** — designed for real-world clinical archives: 2D thick-slice
  acquisitions, anisotropic voxels, and older scans, as well as research-grade
  3D images. Resampling and reorientation happen internally; the output always
  lands back on the input scan's grid.
- Validated in the [Nature Communications study](https://www.nature.com/articles/s41467-025-58274-8)
  on tens of thousands of scans from MS clinical archives and trials, where it
  measured established treatment effects from scans conventional pipelines
  cannot process.

**Intended use**: research only. MindGlide is not a medical device and must not
be used for clinical decision-making.

**Speed** (measured): seconds per scan on a modern CUDA GPU (~10 s including
model load on a 2016-era Quadro P6000); ~1.5 min for a 2 mm scan and a few
minutes for a 1 mm scan on a multi-core CPU.

<details>
<summary><strong>Troubleshooting & FAQ</strong></summary>

**Do I need to skull-strip / bias-correct / register first?** — No. Feed the
raw NIfTI.

**"Warning: not using the GPU — … this PyTorch build cannot run on it"** —
the default `pip` PyTorch wheels no longer include kernels for older GPUs
(e.g. Pascal cards: GTX 10xx, Quadro P series). MindGlide falls back to CPU
automatically. To use such a GPU, install a compatible PyTorch first:

```bash
pip install "torch==2.6.0+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install mindglide
```

**GPU out of memory** — try `--sw-batch-size 1`, or `--device cpu`.

**Apple Silicon** — `auto` uses MPS when available. If an operation is
unsupported, run with `--device cpu` or set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Offline / air-gapped machines** — download
[the checkpoint](https://huggingface.co/MS-PINPOINT/mindglide/tree/main) once
and pass `--model-path /path/to/model.pt` (or set `MODEL_PATH`).

**Model cache location** — the auto-downloaded model lives in the Hugging Face
cache (`~/.cache/huggingface` by default); set `HF_HOME` to move it.

**Can I fine-tune it?** — The original container-based training/fine-tuning
pipeline is preserved at the
[`legacy-container`](https://github.com/MS-PINPOINT/mindGlide/tree/legacy-container)
tag. Open a [Discussion](https://github.com/MS-PINPOINT/mindGlide/discussions)
if you're interested.

</details>

<details>
<summary><strong>Docker / Apptainer</strong></summary>

Prebuilt images (model weights baked in — works offline; ~8 GB with the CUDA
runtime) are published on every release:

```bash
docker pull ghcr.io/ms-pinpoint/mindglide:latest

# run on a folder ( --user keeps output files owned by you; drop --gpus all on CPU-only hosts )
docker run --gpus all --ipc=host --user $(id -u):$(id -g) -v /data:/data \
  ghcr.io/ms-pinpoint/mindglide:latest -i /data/scan.nii.gz -o /data/scan_seg.nii.gz
```

For Apptainer/Singularity on HPC:

```bash
apptainer pull mindglide.sif docker://ghcr.io/ms-pinpoint/mindglide:latest
apptainer run --nv -B /data:/data mindglide.sif -i /data/scan.nii.gz -o /data/scan_seg.nii.gz
```

To build the image yourself instead: `git clone` this repo and
`docker build -t mindglide .`

</details>

<details>
<summary><strong>Model weights</strong></summary>

The checkpoint (`_20240404_conjurer_trained_dice_7733.pt`) is downloaded
automatically from
[Hugging Face: MS-PINPOINT/mindglide](https://huggingface.co/MS-PINPOINT/mindglide)
on first run, pinned to an exact revision for reproducibility. Additional and
legacy checkpoints are archived in the same repository. Models were trained on
the datasets described in the
[paper](https://www.nature.com/articles/s41467-025-58274-8).

From a source checkout you can also fetch the weights as a git submodule
(requires [Git LFS](https://git-lfs.com)):

```bash
git submodule update --init --recursive
git submodule foreach 'git lfs pull'
```

</details>

<details>
<summary><strong>Development & tests</strong></summary>

```bash
git clone https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
pip install -e ".[test]"

pytest                          # fast unit tests (seconds, no model download)
MINDGLIDE_RUN_SLOW=1 pytest -v  # + end-to-end on a public MNI scan (CPU, and GPU if present)
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes to the numerical path must
produce byte-identical segmentations (the e2e tests check real outputs on
public data).

</details>

## Citation

If you use MindGlide, please cite (or use GitHub's *Cite this repository*
button):

> Goebl P, Wingrove J, Abdelmannan O, *et al.* Enabling new insights from old
> scans by repurposing clinical MRI archives for multiple sclerosis research.
> *Nature Communications*. 2025;16(1):3149. doi:10.1038/s41467-025-58274-8

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

This study and the initial development of this project were funded by the UK National Institute for Health and Care Research (NIHR) Advanced Fellowship to Arman Eshaghi (Award ID: NIHR302495). The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.

As of 2026, the team and this research have transitioned to the **DreaMS Lab** at **King's College London** (Institute of Psychiatry, Psychology & Neuroscience), backed by an 8-year project funded by **Wellcome**.

<p align="left">
  <img src="https://raw.githubusercontent.com/MS-PINPOINT/mindGlide/main/assets/nihr_logo.png" alt="NIHR logo" width="200">
</p>
