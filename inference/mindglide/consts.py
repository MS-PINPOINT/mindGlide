PATCH_SIZE          = [128, 128, 64]
SPACING             = [1.0, 1.0, 1.0]
CLIP_VALUES         = [0, 0]
NORMALIZE_VALUES    = [0, 1]
DEEP_SUPR_NUM       = 3


PROPERTIES = {'name': 'MindGlide',
              'description': 'segments any modality into CGM, DGM and a few others',
              'reference': 'MindGlide',
              'licence': 'CC-BY-SA 4.0',
              'release': '1.0 01/03/2023',
              'tensorImageSize': '3D',
              'modality': {'0': 'receives only one and any modality.'},
              'labels': {'0': 'Background',
                         '1': 'CSF',
                         '2': 'Ventricles_3_4_5',
                         '3': 'DGM',
                         '4': 'Pons',
                         '5': 'Brainstem',
                         '6': 'Cerebellum',
                         '7': 'Temporal_lobe',
                         '8': 'Temporal_horn_lateral_ventricle',
                         '9': 'Lateral_ventricle',
                         '10': 'Optic_chiasm',
                         '11': 'Cerebellar_vermis',
                         '12': 'Corpus_callosum',
                         '13': 'White_matter',
                         '14': 'Frontal_lobe_GM',
                         '15': 'Limbic_cortex_GM',
                         '16': 'Parietal_lobe_GM',
                         '17': 'Occipital_lobe_GM',
                         '18': 'Lesion',
                         '19': 'Ventral_diencephalon'},
                    'numTest': 1,
                    'numTraining': 0,
}