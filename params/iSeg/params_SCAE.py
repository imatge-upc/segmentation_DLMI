import params as p
from src.config import DB


def get_params():
    params = {
        p.DATABASE: 'iSeg',
        p.INPUT_DIM: (64,64,64),

        p.N_CLASSES: 4,
        p.N_EPOCHS: 150,
        p.N_SUBEPOCHS: 1,
        p.BATCH_SIZE: 10,
        p.CLASS_WEIGHTS: 'inverse_weights',

        p.SAMPLING_SCHEME: 'uniform',
        p.SAMPLING_WEIGHTS: [0.5, 0.5],
        p.N_SEGMENTS_TRAIN: 70,
        p.N_SUBJECTS_TRAIN: 7,

        p.N_SEGMENTS_VALIDATION: 30,
        p.N_SUBJECTS_VALIDATION: 3,

        p.TRAIN_SIZE: 0.7,
        p.DEV_SIZE: 0.3,

        p.DATA_AUGMENTATION_FLAG: False,

        p.OUTPUT_PATH: '/work/acasamitjana/segmentation/iSeg/SCAE',
        p.MODEL_NAME: 'stacked_CAE_masked',
        p.LR: 0.005

    }

    return params