import params as p
from src.config import DB

# IMPORTANT! The criteria used to choose the parameters is that there are 3 samples of +/- per subject
# in every subepoch. There are 600 segments / subepoch.
# 1*5*20*100 = 10000 ( N_EXAMPLES_PER_CLASS * N_CLASSES * N_SUBEPOCHS * N_SUBJECTS_TRAIN = N_SEGMENTS_TRAIN )
def get_params():
    params = {
        p.DATABASE: 'BraTS2017',
        p.INPUT_DIM: [192,192,160],#[64,64,64],#

        p.N_CLASSES: 3,
        p.N_EPOCHS: 50,
        p.N_SUBEPOCHS: None,#1,#
        p.BATCH_SIZE: 1,#10,#
        p.CLASS_WEIGHTS: 'inverse_weights',

        p.SAMPLING_SCHEME: 'whole',#'whole',#
        p.SAMPLING_WEIGHTS: None,
        p.N_SEGMENTS_TRAIN: None,#120,#
        p.N_SUBJECTS_TRAIN: 200,#None,#

        p.N_SEGMENTS_VALIDATION: None,#10,#
        p.N_SUBJECTS_VALIDATION: 50,#None, #

        p.TRAIN_SIZE: 0.6,
        p.DEV_SIZE: 0.4,

        p.DATA_AUGMENTATION_FLAG: ['saggital-planes','coronal-plane','axial-plane'],
        p.NUM_MODALITIES: 4,

        p.MODEL_TYPE: 'mask_seg',
        p.LOSS: 'xentropy_sw',

        p.SAMPLE_WEIGHTS_BOOL: True,

        p.OUTPUT_PATH: '/work/acasamitjana/segmentation/BraTS/mask_seg/20170819',
        p.MODEL_NAME: 'brats2017',
        p.LR: 0.0001
    }

    return params

