import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from os.path import join
from keras import backend as K
import params.BraTS as p
from src.config import DB
from database.BraTS.data_loader import Loader
from src.dataset import Dataset_train
from src.models import SegmentationModels
from src.utils import io, preprocessing
from src.metrics_old import dice_whole, dice_core, dice_enhance, categorical_crossentropy_3d
import nibabel as nib

if __name__ == "__main__":

    K.set_image_data_format('channels_first')

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    filename = params[p.MODEL_NAME]
    dir_path = join(params[p.OUTPUT_PATH])

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
        num_classes=5,
        model_name=params[p.MODEL_NAME],
    )
    #
    model.load_weights(weights_filepath)
    model.compile(optimizer='Adam', loss=categorical_crossentropy_3d,metrics = [dice_whole,dice_enhance,dice_core])
    #
    # model.summary()

    print("Architecture defined ...")
    """ DATA LOADING """
    brats_db = Loader.create(config_dict=DB.BRATS2016)
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
    generator_train = dataset.data_generator_full(subject_list_train, mode='validation', data_format = 'channels_first')


    n_sbj = 0
    for image,labels in generator_train:

        predictions = model.predict_on_batch(image)

        predictions = np.transpose(predictions, (0, 2, 3, 4, 1))
        labels = np.transpose(labels, (0, 2, 3, 4, 1))

        print(predictions.shape)
        predictions = np.concatenate([
            predictions[:, :, :, :, 0:1],
            np.sum(np.concatenate([predictions[:, :, :, :, 1:2], predictions[:, :, :, :, 3:4]],axis=4), axis=4, keepdims=True),
            predictions[:, :, :, :, 2:3],
            predictions[:, :, :, :, 4:]
        ],axis=4)
        print(predictions.shape)
        for i in range(4):
            tmp = predictions[:,:,:,:,i]
            print(np.sort(tmp.flatten())[::-1])

        img = nib.Nifti1Image(preprocessing.resize_image(np.argmax(predictions,axis=4)[0],(240,240,155)), subject_list[0].get_affine())
        nib.save(img, join(dir_path, 'results', str(subject_list_train[n_sbj].id) + '_predictions.nii.gz'))

        img = nib.Nifti1Image(preprocessing.resize_image(np.argmax(labels,axis=4)[0],(240,240,155)), subject_list[0].get_affine())
        nib.save(img, join(dir_path, 'results', str(subject_list_train[n_sbj].id) + '_labels.nii.gz'))
        print('Subject ' + str(subject_list_train[n_sbj].id) + ' has finished')
        n_sbj +=1
        if n_sbj>5:
            break

    metrics = model.evaluate_generator(generator_train, steps=len(subject_list_train))
    print(metrics)

    print('Finished training')
