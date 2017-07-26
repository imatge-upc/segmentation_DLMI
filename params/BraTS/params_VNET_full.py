import params as p
from src.config import DB

# IMPORTANT! The criteria used to choose the parameters is that there are 3 samples of +/- per subject
# in every subepoch. There are 600 segments / subepoch.
# 1*5*20*100 = 10000 ( N_EXAMPLES_PER_CLASS * N_CLASSES * N_SUBEPOCHS * N_SUBJECTS_TRAIN = N_SEGMENTS_TRAIN )
def get_params():
    params = {
        p.DATABASE: 'BraTS2017',
        p.INPUT_DIM: [192,192,160],#[64,64,64],#

        p.N_CLASSES: 4,
        p.N_EPOCHS: 150,
        p.N_SUBEPOCHS: 1,#20,#
        p.BATCH_SIZE: 1,#10,#
        p.CLASS_WEIGHTS: 'inverse_weights',

        p.SAMPLING_SCHEME: 'whole',#'foreground-background',#
        p.SAMPLING_WEIGHTS: [0.5, 0.5],
        p.N_SEGMENTS_TRAIN: 120,#4000,#
        p.N_SUBJECTS_TRAIN: None,#100,#

        p.N_SEGMENTS_VALIDATION: 10,#400,#
        p.N_SUBJECTS_VALIDATION: None, #80,#

        p.TRAIN_SIZE: 0.6,
        p.DEV_SIZE: 0.4,

        p.DATA_AUGMENTATION_FLAG: False,
        p.NUM_MODALITIES: 4,

        p.BN_LAST: False,
        p.SHORTCUT_INPUT: False,
        p.BOOL_MASK: True,

        p.OUTPUT_PATH: '/work/acasamitjana/segmentation/BraTS/brats2017',
        p.MODEL_NAME: 'v_net_BN',
        p.LR: 0.01
    }

    return params

