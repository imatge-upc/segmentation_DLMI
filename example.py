import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join

import params as p
from params.params_example import get_params
from src.config import DB
from database.WMH.data_loader import Loader
from src.dataset import Dataset_train
from src.models import SegmentationModels
from src.utils import io, preprocessing
import nibabel as nib

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() + np.finfo('float').eps) / (im1.sum() + im2.sum() + np.finfo('float').eps)




params = get_params()

print("Architecture defined ...")
""" DATA LOADING """
wmh_db = Loader.create(config_dict=DB.WMH)
subject_list = wmh_db.load_subjects()

dataset = Dataset_train(input_shape=tuple(params[p.INPUT_DIM]),
                        output_shape=tuple(params[p.INPUT_DIM]),
                        n_classes=params[p.N_CLASSES],
                        n_subepochs=params[p.N_SUBEPOCHS],
                        batch_size=1,

                        sampling_scheme=params[p.SAMPLING_SCHEME],
                        sampling_weights=params[p.SAMPLING_WEIGHTS],

                        n_subjects_per_epoch_train=params[p.N_SUBJECTS_TRAIN],
                        n_segments_per_epoch_train=params[p.N_SEGMENTS_TRAIN],

                        n_subjects_per_epoch_validation=params[p.N_SUBJECTS_VALIDATION],
                        n_segments_per_epoch_validation=params[p.N_SEGMENTS_VALIDATION],

                        train_size=params[p.TRAIN_SIZE],
                        dev_size=params[p.DEV_SIZE],
                        num_modalities=params[p.NUM_MODALITIES],

                        data_augmentation_flag= params[p.DATA_AUGMENTATION_FLAG],
                        class_weights=params[p.CLASS_WEIGHTS]
                        )


print('TrainVal Dataset initialized')
subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)

""" ARCHITECTURE DEFINITION """

model, output_shape = SegmentationModels.get_model(
    num_modalities=params[p.NUM_MODALITIES],
    segment_dimensions=tuple(params[p.INPUT_DIM]),
    num_classes=params[p.N_CLASSES],
    model_name=params[p.MODEL_NAME],
    shortcut_input = params[p.SHORTCUT_INPUT]
)

model = SegmentationModels.compile(model, lr=params[p.LR], num_classes=params[p.N_CLASSES])

model.summary()


""" MODEL TESTING """
print()
print('Testint started...')
print('Output_shape: ' + str(output_shape))

subject_list_train = subject_list_train
generator_train = dataset.data_generator_full_mask(subject_list_train, mode='validation')
generator_val = dataset.data_generator_full_mask(subject_list_validation, mode='validation')

steps_per_epoch = int(len(subject_list_train)) if params[p.N_SUBJECTS_TRAIN] is None else params[p.N_SUBJECTS_TRAIN]
validation_steps = int(len(subject_list_validation)) if params[p.N_SUBJECTS_VALIDATION] is None else params[p.N_SUBJECTS_VALIDATION]


n_sbj = 0
dice_1 = np.zeros(len(subject_list_validation))
dice_2 = np.zeros(len(subject_list_validation))

inputs, outputs = next(generator_train)
inputs_val, outputs_val = next(generator_val)

hist = model.fit(inputs, outputs,
                 batch_size = params[p.BATCH_SIZE],
                 epochs = params[p.N_EPOCHS],
                 validation_data = (inputs_val, outputs_val))

metrics = model.evaluate(inputs_val, outputs_val, batch_size =params[p.BATCH_SIZE])

np.testing.assert_almost_equal(metrics[0], hist.history['val_loss'], decimal = 4)
# print(hist.history)
# print(metrics)
print('Test OK')
print('')
print('')

print('Example finished')