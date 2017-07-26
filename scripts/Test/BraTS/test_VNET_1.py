import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join

import params.BraTS as p
from src.config import DB
from database.BraTS.data_loader_tumor_mask import Loader
from src.dataset import Dataset_train
from src.models import SegmentationModels
from src.utils import io, preprocessing
import nibabel as nib

if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    filename = 'v_net'#params[p.MODEL_NAME]
    dir_path = join(params[p.OUTPUT_PATH], 'LR_' + str(params[p.LR])+ '_patches_xentropy')

    logs_filepath = join(dir_path, 'logs', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')


    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    # io.redirect_stdout_to_file(filepath=logs_filepath)




    """ ARCHITECTURE DEFINITION """
    model, output_shape = SegmentationModels.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=(192,192,160),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        BN_last=params[p.BN_LAST],
        shortcut_input = params[p.SHORTCUT_INPUT],
        mask_bool=params[p.BOOL_MASK]
    )
    #
    model.load_weights(weights_filepath)
    model = SegmentationModels.compile(model, lr=params[p.LR],num_classes=params[p.N_CLASSES])
    #
    # model.summary()

    print("Architecture defined ...")
    """ DATA LOADING """
    brats_db = Loader.create(config_dict=DB.BRATS2017)
    subject_list = brats_db.load_subjects()

    dataset = Dataset_train(input_shape=(192,192,160),
                            output_shape=(192,192,160),
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

    print('TrainVal Dataset initialized')
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)


    """ MODEL TRAINING """
    print()
    print('Training started...')
    print('Steps per epoch: ' + str(params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE]))
    print('Output_shape: ' + str(output_shape))


    subject_list_train = subject_list_train
    generator_train = dataset.data_generator_full_mask(subject_list_train, mode='validation')


    n_sbj = 0
    for image,labels in generator_train:
        predictions = model.predict_on_batch(image)

        img = nib.Nifti1Image(np.argmax(predictions,axis=4)[0], subject_list[0].get_affine())
        nib.save(img, join(dir_path, 'results', str(subject_list_train[n_sbj].id) + '_predictions.nii.gz'))

        img = nib.Nifti1Image(np.argmax(labels,axis=4)[0], subject_list[0].get_affine())
        nib.save(img, join(dir_path, 'results', str(subject_list_train[n_sbj].id) + '_labels.nii.gz'))
        print('Subject ' + str(subject_list_train[n_sbj].id) + ' has finished')
        n_sbj +=1
        if n_sbj>5:
            break

    metrics = model.evaluate_generator(generator_train, steps=len(subject_list_train))
    print(metrics)

    print('Finished training')
