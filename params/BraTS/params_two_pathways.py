import params as p
from src.config import DB

# IMPORTANT! The criteria used to choose the parameters is that there are 3 samples of +/- per subject
# in every subepoch. There are 600 segments / subepoch.
# 1*5*20*100 = 10000 ( N_EXAMPLES_PER_CLASS * N_CLASSES * N_SUBEPOCHS * N_SUBJECTS_TRAIN = N_SEGMENTS_TRAIN )
def get_params():
    params = {
        p.DATABASE: 'BraTS2017',
        p.INPUT_DIM: [224,224,160],#[64,64,64],#

        p.N_CLASSES: 4,
        p.N_EPOCHS: 150,
        p.N_SUBEPOCHS: 20,#1,#
        p.BATCH_SIZE: 1,#10,#
        p.CLASS_WEIGHTS: 'inverse_weights',

        p.SAMPLING_SCHEME: 'foreground-background',#'whole',#
        p.SAMPLING_WEIGHTS: [0.5, 0.5],
        p.N_SEGMENTS_TRAIN: 400,#120,#
        p.N_SUBJECTS_TRAIN: 100,#None,#

        p.N_SEGMENTS_VALIDATION: 80,#10,#
        p.N_SUBJECTS_VALIDATION: 20,#None, #

        p.TRAIN_SIZE: 0.6,
        p.DEV_SIZE: 0.4,

        p.DATA_AUGMENTATION_FLAG: False,
        p.NUM_MODALITIES: 4,

        p.BN_LAST: False,
        p.SHORTCUT_INPUT: False,
        p.BOOL_MASK: False,

        p.OUTPUT_PATH: '/work/acasamitjana/segmentation/BraTS/two_pathways',
        p.MODEL_NAME: 'two_pathways',
        p.LR: 0.0005
    }

    return params

