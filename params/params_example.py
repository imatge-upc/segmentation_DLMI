import params as p
from src.config import DB

# IMPORTANT! The criteria used to choose the parameters is that there are 3 samples of +/- per subject
# in every subepoch. There are 600 segments / subepoch.
# 1*5*20*100 = 10000 ( N_EXAMPLES_PER_CLASS * N_CLASSES * N_SUBEPOCHS * N_SUBJECTS_TRAIN = N_SEGMENTS_TRAIN )
def get_params():
    params = {
        p.DATABASE: DB.WMH,
        p.INPUT_DIM: [192,192,96],#[32,32,32],

        p.N_CLASSES: 3,
        p.N_EPOCHS: 1,
        p.N_SUBEPOCHS: 1,
        p.BATCH_SIZE: 1,
        p.CLASS_WEIGHTS: 'inverse_weights',

        p.SAMPLING_SCHEME: 'whole',
        p.SAMPLING_WEIGHTS: [0.5, 0.5],
        p.N_SEGMENTS_TRAIN: 720,
        p.N_SUBJECTS_TRAIN: 1,

        p.N_SEGMENTS_VALIDATION: 240,
        p.N_SUBJECTS_VALIDATION: 1,

        p.TRAIN_SIZE: 0.6,
        p.DEV_SIZE: 0.4,

        p.DATA_AUGMENTATION_FLAG: True, #It can bee False, True (=saggital-plane), saggital_plane or all-planes
        p.NUM_MODALITIES: 2,


        p.SHORTCUT_INPUT: True,

        p.OUTPUT_PATH: '/',
        p.MODEL_NAME: 'Masked_v_net_BN',
        p.LR: 0.0005
    }

    return params

