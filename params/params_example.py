import params as p


def get_params():
    params = {
        p.INPUT_DIM: [64,64,64],
        p.DATA_PATH: ['/projects/neuro/BRATS/BRATS2015_Training/HGG',
                      '/projects/neuro/BRATS/BRATS2015_Training/LGG'],
        p.N_CLASSES: 5,
        p.N_SUBEPOCHS: 20,
        p.BATCH_SIZE: 5,
        p.CLASS_WEIGHTS: 'inverse_weights',  # None or 'constant': w=1, 'inverse_weights' w [prop] 1/f_i

        p.SAMPLING_SCHEME: 'foreground-background',
        p.SAMPLING_WEIGHTS: [0.5, 0.5],
        p.N_SEGMENTS_TRAIN: 8000,
        p.N_SUBJECTS_TRAIN: 100,

        p.N_SEGMENTS_VALIDATION: 600,
        p.N_SUBJECTS_VALIDATION: 100,

        p.ID_LIST_TRAIN: 0.6,
        p.ID_LIST_VALIDATION: 0.4,

        p.BOOLEAN_FLAIR: True,
        p.BOOLEAN_T1: True,
        p.BOOLEAN_T1C: True,
        p.BOOLEAN_T2: True,
        p.BOOLEAN_ROImask: True,
        p.BOOLEAN_LABELS: True,

        p.DATA_AUGMENTATION_FLAG: False,

        p.OUTPUT_PATH: '/imatge/acasamitjana/work/BRATS/output',

    }

    return params