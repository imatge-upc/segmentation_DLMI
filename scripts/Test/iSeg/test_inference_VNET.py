import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join, exists
import os
import params.iSeg as p
from src.config import DB
from database.iSeg.data_loader import Loader
from src.dataset import Dataset_train
from src.models import iSeg_models
from src.utils import io, preprocessing
import nibabel as nib

def tf_labels(labels):
    labels[np.where(labels == 1)] = 10
    labels[np.where(labels == 2)] = 150
    labels[np.where(labels == 3)] = 250

    return labels



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

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    params[p.INPUT_DIM] = (192,192,144)
    params[p.BATCH_SIZE] = 1


    filename = params[p.MODEL_NAME]
    dir_path = join(params[p.OUTPUT_PATH], 'LR_' + str(params[p.LR])+'_full_DA_shortcutTrue')

    logs_filepath = join(dir_path, 'logs', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')


    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    # io.redirect_stdout_to_file(filepath=logs_filepath)




    """ ARCHITECTURE DEFINITION """
    num_modalities = 2
    model, output_shape = iSeg_models.get_model(
        num_modalities=num_modalities,
        segment_dimensions=params[p.INPUT_DIM],
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        shortcut_input=[p.SHORTCUT_INPUT],
        mode='test'
    )

    model.load_weights(weights_filepath)
    model = iSeg_models.compile(model, lr=params[p.LR],num_classes=params[p.N_CLASSES])

    model.summary()

    print("Architecture defined ...")
    """ DATA LOADING """
    iseg_db = Loader.create(config_dict=DB.iSEG_test)
    subject_list = iseg_db.load_subjects()

    dataset = Dataset_train(input_shape=params[p.INPUT_DIM],
                            output_shape=params[p.INPUT_DIM],
                            n_classes=params[p.N_CLASSES],
                            n_subepochs=params[p.N_SUBEPOCHS],
                            batch_size=params[p.BATCH_SIZE],

                            sampling_scheme=params[p.SAMPLING_SCHEME],
                            sampling_weights=params[p.SAMPLING_WEIGHTS],

                            n_subjects_per_epoch_train=params[p.N_SUBJECTS_TRAIN],
                            n_segments_per_epoch_train=params[p.N_SEGMENTS_TRAIN],

                            n_subjects_per_epoch_validation=params[p.N_SUBJECTS_VALIDATION],
                            n_segments_per_epoch_validation=params[p.N_SEGMENTS_VALIDATION],

                            train_size=params[p.TRAIN_SIZE],
                            dev_size=params[p.DEV_SIZE],
                            num_modalities=num_modalities,

                            data_augmentation_flag= params[p.DATA_AUGMENTATION_FLAG],
                            class_weights=params[p.CLASS_WEIGHTS]
                            )


    """ MODEL TRAINING """
    print('')
    print('Training started...')
    print('Steps per epoch: ' + str(params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE]))
    print('Output_shape: ' + str(output_shape))

    generator = dataset.data_generator_inference(subject_list, normalize_bool=True)

    n_sbj = 0


    for inputs in generator:
        subject = subject_list[n_sbj]
        print(subject.id)

        predictions = model.predict_on_batch(inputs)[0]

        shape = subject.get_subject_shape()
        predictions_resized = np.zeros(shape+(params[p.N_CLASSES],))
        for i in range(params[p.N_CLASSES]):
            predictions_resized[:,:,:,i] = preprocessing.resize_image(predictions[:,:,:,i],shape)

        #Saving images
        predictions_resize_argmax = np.argmax(predictions_resized,axis=3)
        predictions_argmax = np.argmax(predictions, axis=3)

        if not exists(join(dir_path, 'results_test')):
            os.makedirs(join(dir_path, 'results_test'))

        img = nib.Nifti1Image(tf_labels(predictions_resize_argmax), subject.get_affine())
        nib.save(img, join(dir_path, 'results_test', 'subject-' + subject.id + '-label.nii.gz'))
        n_sbj += 1
        if n_sbj >= len(subject_list):
            break



    print('Finished testing')
