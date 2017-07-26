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
from src.models import SegmentationModels
from src.utils import io
import nibabel as nib
if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    filename = params[p.MODEL_NAME]
    dir_path = join(params[p.OUTPUT_PATH],
                    'LR_' + str(params[p.LR]) + '_dice_DA' + str(params[p.DATA_AUGMENTATION_FLAG]) + '_BN_'
                    + str(params[p.BN_LAST]) + '_shortcut_' + str(params[p.SHORTCUT_INPUT]) + '_mask_' + str(
                        params[p.BOOL_MASK]))

    logs_filepath = join(dir_path, 'logs', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')


    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    # io.redirect_stdout_to_file(filepath=logs_filepath)



    print("Architecture defined ...")
    """ DATA LOADING """
    wmh_db = Loader.create(config_dict=DB.WMH)
    subject_list = wmh_db.load_subjects()
    num_modalities = 2


    max_shape = [0,0,0]
    for subject in subject_list:
        for it_dim,dim in  enumerate(subject.get_subject_shape()):
            if dim > max_shape[it_dim]:
                print(dim)
                max_shape[it_dim] = dim

    max_shape = [m_s + np.mod(32-np.mod(m_s,32),32) for m_s in max_shape]
    print(max_shape)

    dataset = Dataset_train(input_shape=tuple(max_shape),
                            output_shape=tuple(max_shape),
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

    model, output_shape = SegmentationModels.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(max_shape),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        BN_last = params[p.BN_LAST],
        shortcut_input = params[p.SHORTCUT_INPUT],
        mask_bool = params[p.BOOL_MASK]
    )

    model.load_weights(weights_filepath, by_name=True)
    model = SegmentationModels.compile(model, lr=params[p.LR], num_classes=params[p.N_CLASSES])

    model.summary()


    """ MODEL TRAINING """
    print()
    print('Training started...')
    print('Steps per epoch: ' + str(params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE]))
    print('Output_shape: ' + str(output_shape))

    subject_list_train = subject_list_train[-10:]
    if params[p.BOOL_MASK]:
        generator_train = dataset.data_generator_full_mask(subject_list_train, mode='validation')
        generator_val = dataset.data_generator_full_mask(subject_list_validation, mode='validation')
    else:
        generator_train = dataset.data_generator_full(subject_list_train, mode='validation')
        generator_val = dataset.data_generator_full(subject_list_validation, mode='validation')

    n_sbj = 0
    for inputs,labels in generator_train:
        subject = subject_list_train[n_sbj]

        predictions = model.predict_on_batch(inputs)
        print(np.unique(np.argmax(predictions[:, :, :, :, :],axis=4)))

        img = nib.Nifti1Image(np.argmax(predictions[:, :, :, :, :],axis=4)[0], subject.get_affine())
        nib.save(img, join(dir_path, 'results', subject.id + '_predictions.nii.gz'))

        img = nib.Nifti1Image(np.argmax(labels[0, :, :, :, :],axis=3), subject.get_affine())
        nib.save(img, join(dir_path, 'results', subject.id + '_labels.nii.gz'))
        print('Subject ' + str(subject.id) + ' has finished')
        n_sbj += 1
        if n_sbj > 5:
            break

    print('Finished training')