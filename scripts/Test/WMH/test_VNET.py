import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join

import params.WMH as p
from src.config import DB
from database.WMH.data_loader import Loader
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


    filename = params[p.MODEL_NAME] + '_continue'
    dir_path = join(params[p.OUTPUT_PATH],
                    'LR_' + str(params[p.LR]) + '_DA_6_4' )

    logs_filepath = join(dir_path, 'results', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')


    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    io.redirect_stdout_to_file(filepath=logs_filepath)



    print("Architecture defined ...")
    """ DATA LOADING """
    wmh_db = Loader.create(config_dict=DB.WMH)
    subject_list = wmh_db.load_subjects()
    num_modalities = 2

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
                            num_modalities=num_modalities,

                            data_augmentation_flag= params[p.DATA_AUGMENTATION_FLAG],
                            class_weights=params[p.CLASS_WEIGHTS]
                            )


    print('TrainVal Dataset initialized')
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)

    """ ARCHITECTURE DEFINITION """

    model, output_shape = WMH_models.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME]+'_old',
        shortcut_input = params[p.SHORTCUT_INPUT],
        mode='test'
    )

    model.load_weights(weights_filepath)
    model = WMH_models.compile(model, lr=params[p.LR], num_classes=params[p.N_CLASSES])

    # model.summary()


    """ MODEL TESTINT """
    print()
    print('Testint started...')
    print('Output_shape: ' + str(output_shape))

    subject_list_train = subject_list_train
    generator_train = dataset.data_generator_full_mask(subject_list_train, mode='validation', normalize_bool=True)
    generator_val = dataset.data_generator_full_mask(subject_list_validation, mode='validation', normalize_bool=True)

    n_sbj = 0
    # metrics = model.evaluate_generator(generator_val, steps=len(subject_list_validation))
    # print(metrics)
    dice_1 = np.zeros(len(subject_list_validation))
    dice_2 = np.zeros(len(subject_list_validation))


    for inputs,labels in generator_val:
        subject = subject_list_validation[n_sbj]

        predictions = model.predict_on_batch(inputs)[0]
        predictions = np.floor((predictions+np.finfo(float).eps)/np.max(predictions,axis=3,keepdims=True)).astype('int')

        shape = subject.get_subject_shape()
        predictions_resized = np.zeros(shape+(params[p.N_CLASSES],))
        for i in range(params[p.N_CLASSES]):
            predictions_resized[:,:,:,i] = preprocessing.resize_image(predictions[:,:,:,i],shape)


        img = nib.Nifti1Image(np.argmax(predictions_resized,axis=3), subject.get_affine())
        nib.save(img, join(dir_path, 'results', subject.id + '_predictions.nii.gz'))

        # img = nib.Nifti1Image(np.argmax(labels[0, :, :, :, :],axis=3), subject.get_affine())
        # nib.save(img, join(dir_path, 'results', subject.id + '_labels.nii.gz'))


        dice_1[n_sbj] = dice(predictions[:,:,:,1].flatten(),labels[0,:,:,:,1].flatten())
        dice_2[n_sbj] = dice(predictions[:,:,:,2].flatten(),labels[0,:,:,:,2].flatten())

        print("Count 1: " + str(np.sum(predictions[:,:,:,1]*labels[0,:,:,:,1])) +' '+ str(np.sum(predictions[:,:,:,1])) + ' ' + str(np.sum(labels[0,:,:,:,1])))
        print("Dice 1: " + str(dice_1[n_sbj]))
        print("Dice 2: " + str(dice_2[n_sbj]))


        print('Subject ' + str(subject.id) + ' has finished')
        n_sbj += 1
        if n_sbj >= len(subject_list_validation):
            break

    print('Average Metrics: ')
    print(dice_1)
    print(dice_2)
    print('Dice_1: ' + str(np.mean(dice_1)))
    print('Dice_2: ' + str(np.mean(dice_2)))




    print('Finished training')