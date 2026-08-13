# mindGlide label colormaps

Colormap / lookup-table files so that mindGlide segmentations (`mindglide -i scan.nii.gz -o seg.nii.gz`) show named, colored regions in standard neuroimaging viewers. All three files encode the same 20-label palette.

## Labels

| Index | Name | R | G | B |
|------:|------|--:|--:|--:|
| 0 | Background | 0 | 0 | 0 |
| 1 | CSF | 108 | 180 | 255 |
| 2 | Ventricles_3_4_5 | 42 | 110 | 235 |
| 3 | DGM | 170 | 80 | 205 |
| 4 | Pons | 255 | 170 | 60 |
| 5 | Brainstem | 230 | 115 | 25 |
| 6 | Cerebellum | 76 | 187 | 100 |
| 7 | Temporal_lobe | 235 | 205 | 60 |
| 8 | Temporal_horn_lateral_ventricle | 90 | 220 | 220 |
| 9 | Lateral_ventricle | 25 | 70 | 190 |
| 10 | Optic_chiasm | 255 | 120 | 185 |
| 11 | Cerebellar_vermis | 25 | 130 | 65 |
| 12 | Corpus_callosum | 205 | 165 | 95 |
| 13 | White_matter | 245 | 245 | 235 |
| 14 | Frontal_lobe_GM | 240 | 140 | 115 |
| 15 | Limbic_cortex_GM | 165 | 110 | 55 |
| 16 | Parietal_lobe_GM | 130 | 90 | 235 |
| 17 | Occipital_lobe_GM | 95 | 200 | 160 |
| 18 | Lesion | 230 | 30 | 30 |
| 19 | Ventral_diencephalon | 255 | 220 | 120 |

## ITK-SNAP (`mindglide_itksnap.label`)

Load your scan and the segmentation, then import the label descriptions:

*Segmentation > Label Editor > Actions > Import label descriptions* and select `mindglide_itksnap.label`.

## FSLeyes (`mindglide_fsleyes.lut`)

Load the segmentation as an overlay, set its overlay type to **Label image**, and select `mindglide_fsleyes.lut` via the **LUT** option in the overlay display panel (choose *Load LUT*). Alternatively, copy the file into your FSLeyes LUT directory (e.g. `~/.config/fsleyes/luts/`) so it appears in the LUT dropdown automatically.

Command line:

```bash
fsleyes scan.nii.gz seg.nii.gz -ot label -l /path/to/mindglide_fsleyes.lut
```

## FreeSurfer / freeview (`mindglide_freesurfer.txt`)

```bash
freeview -v scan.nii.gz seg.nii.gz:colormap=lut:lut=mindglide_freesurfer.txt
```
