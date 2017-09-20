import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join

import params.BraTS as p
from src.config import DB
from database.BraTS.data_loader import Loader
from src.dataset import Dataset_train
from src.models import BraTS_models
from src.utils import io
import nibabel as nib
from src.utils import preprocessing

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

    return 2. * (intersection.sum() + np.finfo('float').eps) / (im1.sum() + im2.sum() + 2*np.finfo('float').eps)



if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    filename = params[p.MODEL_NAME] + '_epoch32'
    dir_path = join(params[p.OUTPUT_PATH], 'LR_' + str(params[p.LR])+'_'+params[p.MODEL_TYPE][0]+'_'+params[p.MODEL_TYPE][1])

    logs_filepath = join(dir_path, 'results_test', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')


    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    io.redirect_stdout_to_file(filepath=logs_filepath)

    print('PARAMETERS')
    print(params)
    print('Learning rate exponential decay')

    """ ARCHITECTURE DEFINITION """
    model1, output_shape = BraTS_models.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        model_type=params[p.MODEL_TYPE][0],
        l1=0.00001,
        l2=0.001,
        momentum=0.99,
        mode='test'
    )
    weights_filepath_1 = '/work/acasamitjana/segmentation/BraTS/mask/20170812/LR_0.0005_mask_dice_momentum0.99/model_weights/brats2017_mask_part.h5'
    model1.load_weights(weights_filepath_1)

    model1 = BraTS_models.compile(model1, lr=params[p.LR], model_type=params[p.MODEL_TYPE][0], loss_name=params[p.LOSS])



    """ ARCHITECTURE DEFINITION """
    model2, output_shape = BraTS_models.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        model_type=params[p.MODEL_TYPE][1],
        l1=0.00001,
        l2=0.001,
        momentum=0.99,
        mode='test'
    )

    weights_filepath_2 = '/imatge/acasamitjana/work/segmentation/BraTS/mask_seg/20170815/LR_0.0005_mask_seg_dice_momentum0.99_brainmask/model_weights/brats2017_epoch32.h5'
    model2.load_weights(weights_filepath_2)
    model2 = BraTS_models.compile(model2, lr=params[p.LR], model_type=params[p.MODEL_TYPE][1], loss_name=params[p.LOSS])


    print("Architecture defined ...")
    """ DATA LOADING """
    brats_db = Loader.create(config_dict=DB.BRATS2017_Validation)
    subject_list = brats_db.load_subjects()

    dataset = Dataset_train(input_shape=tuple(params[p.INPUT_DIM]),
                            output_shape=tuple(params[p.INPUT_DIM]),
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
                            num_modalities=params[p.NUM_MODALITIES],

                            data_augmentation_flag= params[p.DATA_AUGMENTATION_FLAG],
                            class_weights=params[p.CLASS_WEIGHTS]
                            )

    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)

    print("Brats Dataset initialized")

    subject_list_test = subject_list_validation
    # subject_chosen = [subject for subject in subject_list_test if subject.id == 'Brats17_CBICA_BHK_1' ]
    generator_test = dataset.data_generator_BraTS_mask_seg(subject_list_test, mode='test')

    n_sbj = 0
    for inputs, outputs, subject in generator_test:

        labels_mask = outputs[0][0]
        labels_seg = outputs[1][0]

        predictions_mask = model1.predict_on_batch(inputs[:2])[0]
        predictions_seg_tmp = model2.predict_on_batch(inputs[:2])[1][0]


        predictions_mask = np.floor(predictions_mask/np.max(predictions_mask,axis=3,keepdims=True)).astype('int')
        predictions_seg_tmp = np.floor(predictions_seg_tmp/np.max(predictions_seg_tmp,axis=3,keepdims=True)).astype('int')


        predictions_seg = np.zeros(predictions_seg_tmp.shape[:-1]+(predictions_seg_tmp.shape[-1]+1,))
        predictions_seg[:, :, :, 0] = 1 - predictions_mask[:,:,:,1]
        for i in range(predictions_seg.shape[-1]-1):
            predictions_seg[:,:,:,i+1] = predictions_seg_tmp[:,:,:,i]*predictions_mask[:,:,:,1]

        shape = subject.get_subject_shape()

        predictions_mask_resized = np.zeros(shape + (params[p.N_CLASSES],))
        for i in range(2):
            predictions_mask_resized[:,:,:,i] = preprocessing.resize_image(predictions_mask[:,:,:,i],shape)

        predictions_seg_resized = np.zeros(shape + (params[p.N_CLASSES],))
        for i in range(params[p.N_CLASSES]):
            predictions_seg_resized[:,:,:,i] = preprocessing.resize_image(predictions_seg[:,:,:,i],shape)

        #Saving images
        predictions_mask_resize_argmax = np.argmax(predictions_mask_resized,axis=3)
        predictions_seg_resize_argmax = np.argmax(predictions_seg_resized,axis=3)+1

        img = nib.Nifti1Image(predictions_mask_resize_argmax, subject.get_affine())
        nib.save(img, join(dir_path, 'results', subject.id + '_predictions_mask.nii.gz'))

        img = nib.Nifti1Image(predictions_seg_resize_argmax*predictions_mask_resize_argmax, subject.get_affine())
        nib.save(img, join(dir_path, 'results', subject.id + '_predictions_seg.nii.gz'))




        print('Subject ' + str(subject.id) + ' has finished',flush=True)
        n_sbj += 1
        if n_sbj >= len(subject_list_test):
            break


    print('Finished training')
