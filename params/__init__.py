INPUT_DIM = 'input_image_segment_dimensions'
DATA_PATH = 'data_path'
N_CLASSES = 'N_classes'
N_SUBEPOCHS = 'n_subepochs'

SAMPLING_SCHEME = 'Sampling_scheme_used'
SAMPLING_WEIGHTS = "Sampling_weights"

N_SEGMENTS_TRAIN = 'N_segments_per_epoch_train'
N_SUBJECTS_TRAIN = 'N_subjects_per_epoch_train'
ID_LIST_TRAIN = 'Subjects_train'


N_SEGMENTS_VALIDATION = 'N_segments_per_epoch_validation'
N_SUBJECTS_VALIDATION = 'N_subjects_per_epoch_validation'
ID_LIST_VALIDATION = 'Subjects_validation'


BOOLEAN_FLAIR = 'boolean_FLAIR'
BOOLEAN_T1 = 'boolean_T1'
BOOLEAN_T1C = 'booleanT1c'
BOOLEAN_T2 = 'booleanT2'
BOOLEAN_ROImask = 'boolean_ROI_mask'
BOOLEAN_LABELS = 'boolean_labels'

OUTPUT_PATH = 'output_path'
BATCH_SIZE = 'batch_size'

DATA_AUGMENTATION_FLAG = 'data_augmentation'
CLASS_WEIGHTS = 'class_weights'

N_SUBJECTS = 'N_subjects'


from params import params_train, params_pretrain, params_full, params_test
PARAMS_DICT = {
    'pretrain': params_pretrain,
    'train': params_train,
    'full': params_full,
    'test': params_test
}
