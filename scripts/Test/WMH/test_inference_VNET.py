import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join
import os
import params.WMH as p
from src.config import DB
from database.WMH.data_loader_test import Loader
from src.dataset import Dataset_train
from src.models import SegmentationModels, WMH_models
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


if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    params[p.INPUT_DIM] = (192, 192, 96)
    params[p.BATCH_SIZE] = 1
    dir_path = params[p.OUTPUT_PATH]

    weights_filepath = join(os.getcwd(),'scripts','Test','WMH','WMH.h5')

    """ ARCHITECTURE DEFINITION """

    model, output_shape = WMH_models.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        shortcut_input = params[p.SHORTCUT_INPUT],
        mode='test'
    )

    model.load_weights(weights_filepath)
    model = WMH_models.compile(model, lr=params[p.LR], num_classes=params[p.N_CLASSES])
    print("Architecture defined ...")  # model.summary()

    """ DATA LOADING """
    wmh_db = Loader.create(config_dict=DB.WMH)
    subject = wmh_db.load_subject()
    num_modalities = 2

    # dataset = Dataset_train(input_shape=tuple(params[p.INPUT_DIM]),
    #                         output_shape=tuple(params[p.INPUT_DIM]),
    #                         n_classes=params[p.N_CLASSES],
    #                         n_subepochs=params[p.N_SUBEPOCHS],
    #                         batch_size=1,
    #
    #                         sampling_scheme=params[p.SAMPLING_SCHEME],
    #                         sampling_weights=params[p.SAMPLING_WEIGHTS],
    #
    #                         n_subjects_per_epoch_train=params[p.N_SUBJECTS_TRAIN],
    #                         n_segments_per_epoch_train=params[p.N_SEGMENTS_TRAIN],
    #
    #                         n_subjects_per_epoch_validation=params[p.N_SUBJECTS_VALIDATION],
    #                         n_segments_per_epoch_validation=params[p.N_SEGMENTS_VALIDATION],
    #
    #                         train_size=params[p.TRAIN_SIZE],
    #                         dev_size=params[p.DEV_SIZE],
    #                         num_modalities=num_modalities,
    #
    #                         data_augmentation_flag= params[p.DATA_AUGMENTATION_FLAG],
    #                         class_weights=params[p.CLASS_WEIGHTS]
    #                         )



    """ MODEL TESTING """
    print()
    print('Testing started...')
    print('Output_shape: ' + str(output_shape))


    print('Loading...')
    print(subject.data_path)
    shape = subject.get_subject_shape()
    image = subject.load_channels(normalize=True)
    image_resize = np.zeros(params[p.INPUT_DIM]+ (params[p.NUM_MODALITIES],))
    for i in range(params[p.NUM_MODALITIES]):
        image_resize[:,:,:,i] = preprocessing.resize_image(image[:,:,:,i],params[p.INPUT_DIM],pad_value=image[0, 0, 0, i])

    print('mask0')
    mask = preprocessing.resize_image(subject.load_ROI_mask(),params[p.INPUT_DIM])

    print('mask')


    image_resize = image_resize[np.newaxis,:]

    mask_complete = mask[np.newaxis,:,:,:,np.newaxis]


    print('Predicting...')
    prediction = model.predict_on_batch([image_resize,mask_complete])[0]
    prediction = np.floor((prediction + np.finfo(float).eps) / np.max(prediction, axis=3, keepdims=True)).astype('int')

    prediction_resized = preprocessing.resize_image(np.argmax(prediction,axis=3),shape)
    img = nib.Nifti1Image(prediction_resized, subject.get_affine())
    nib.save(img, join(dir_path, 'result.nii.gz'))


    print('Finished testing')
