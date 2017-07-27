import params as p
from src.config import DB

# IMPORTANT! The criteria used to choose the parameters is that there are 3 samples of +/- per subject
# in every subepoch. There are 600 segments / subepoch.
# 1*5*20*100 = 10000 ( N_EXAMPLES_PER_CLASS * N_CLASSES * N_SUBEPOCHS * N_SUBJECTS_TRAIN = N_SEGMENTS_TRAIN )
def get_params():
    params = {
        p.DATABASE: DB.WMH,
        p.INPUT_DIM: [64,64,16],#

        p.N_CLASSES: 3,
        p.N_EPOCHS: 250,
        p.N_SUBEPOCHS: 1,
        p.BATCH_SIZE: 10,
        p.CLASS_WEIGHTS: 'inverse_weights',

        p.SAMPLING_SCHEME: 'foreground-background',
        p.SAMPLING_WEIGHTS: [0.6, 0.4],
        p.N_SEGMENTS_TRAIN: 720,
        p.N_SUBJECTS_TRAIN: None,

        p.N_SEGMENTS_VALIDATION: 240,
        p.N_SUBJECTS_VALIDATION: None,

        p.TRAIN_SIZE: 0.6,
        p.DEV_SIZE: 0.4,

        p.DATA_AUGMENTATION_FLAG: True,
        p.NUM_MODALITIES: 2,


        p.SHORTCUT_INPUT: True,

        p.OUTPUT_PATH: '/work/acasamitjana/segmentation/WMH/20170719/VNet_patches',
        p.MODEL_NAME: 'v_net_BN_patches_sr',
        p.LR: 0.0005
    }

    return params

