import argparse
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from keras import backend as K
from os.path import join

import params.iSeg as p
from src.config import DB
from database.iSeg.data_loader import Loader
from src.dataset import Dataset_train
from src.models import SCAE
from src.utils import io


if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    parameters = p.PARAMS_DICT[params_string].get_params()
    filename = parameters[p.MODEL_NAME]
    dir_path = join(parameters[p.OUTPUT_PATH], 'LR_' + str(parameters[p.LR]))

    logs_filepath = join(dir_path, 'logs', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')


    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    io.redirect_stdout_to_file(filepath=logs_filepath)

    print("Architecture defined ...")
    """ DATA LOADING """
    whm_db = Loader.create(config_dict=DB.iSEG)
    subject_list = whm_db.load_subjects()

    stddev = 0.01
    dataset = Dataset_train(input_shape=tuple(parameters[p.INPUT_DIM]),
                            output_shape=tuple(parameters[p.INPUT_DIM]),
                            n_classes=parameters[p.N_CLASSES],
                            n_subepochs=parameters[p.N_SUBEPOCHS],
                            batch_size=parameters[p.BATCH_SIZE],

                            sampling_scheme=parameters[p.SAMPLING_SCHEME],
                            sampling_weights=parameters[p.SAMPLING_WEIGHTS],

                            n_subjects_per_epoch_train=parameters[p.N_SUBJECTS_TRAIN],
                            n_segments_per_epoch_train=parameters[p.N_SEGMENTS_TRAIN],

                            n_subjects_per_epoch_validation=parameters[p.N_SUBJECTS_VALIDATION],
                            n_segments_per_epoch_validation=parameters[p.N_SEGMENTS_VALIDATION],

                            train_size=parameters[p.TRAIN_SIZE],
                            dev_size=parameters[p.DEV_SIZE],
                            num_modalities=parameters[p.N_CLASSES],

                            data_augmentation_flag= parameters[p.DATA_AUGMENTATION_FLAG],
                            class_weights=parameters[p.CLASS_WEIGHTS]
                            )

    print('TrainVal Dataset initialized')
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)


    """ MODEL TRAINING """
    print('')
    print('Training started...')
    print('Samples per epoch:' + str(len(subject_list_train)))

    CAE_list = []
    for i_CAE in range(5):
        CAE_list.append({'n_filters': 8, 'train_flag': True, 'weights_filename': None})

    # CAE_list[0]['train_flag'] = False
    # CAE_list[0]['weights_filename'] = join(dir_path, 'model_weights', parameters[p.MODEL_NAME] + '_' + str(0) + '.h5')
    #
    # CAE_list[1]['train_flag'] = False
    # CAE_list[1]['weights_filename'] = join(dir_path, 'model_weights', parameters[p.MODEL_NAME] + '_' + str(1) + '.h5')
    # #
    # CAE_list[2]['train_flag'] = False
    # CAE_list[2]['weights_filename'] = join(dir_path, 'model_weights', parameters[p.MODEL_NAME] + '_' + str(2) + '.h5')
    #
    # CAE_list[3]['train_flag'] = False
    # CAE_list[3]['weights_filename'] = join(dir_path, 'model_weights', parameters[p.MODEL_NAME] + '_' + str(3) + '.h5')
    #


    for i_CAE in np.arange(0,len(CAE_list)):
        print('Training CAE ' + str(i_CAE) )
        weights_filepath = join(dir_path, 'model_weights',parameters[p.MODEL_NAME] + '_' + str(i_CAE) + '.h5')

        ''' DATA GENERATOR DEFINITION'''
        CAE_shape = parameters[p.INPUT_DIM]
        num_modalities = 2

        generator_train = dataset.data_generator(subject_list_train)
        generator_val = dataset.data_generator(subject_list_validation)

        """ ARCHITECTURE DEFINITION """
        print('Defining architecture... ', flush=True)
        model = SCAE.get_model(
            num_modalities=parameters[p.N_CLASSES],
            segment_dimensions=CAE_shape,
            model_name=parameters[p.MODEL_NAME],
            CAE=CAE_list[:i_CAE+1],
            stddev = stddev,
            pre_train_CAE=True,
        )
        model = SCAE.compile_scae(model, lr=parameters[p.LR])
        print('Architeture defined: ')

        model.summary()

        json_string = model.to_json()
        open(join(dir_path, parameters[p.MODEL_NAME] + '_' + str(i_CAE) + '_model.json'), 'w').write(
            json_string)


        cb_saveWeights = ModelCheckpoint(filepath=weights_filepath, save_best_only=True, mode='min')

        model.fit_generator(generator_train,
                            steps_per_epoch=int(np.ceil(len(subject_list_train)/parameters[p.BATCH_SIZE])),
                            epochs=parameters[p.N_EPOCHS],
                            validation_data=generator_val,
                            validation_steps=int(np.ceil(len(subject_list_validation)/parameters[p.BATCH_SIZE])),
                            callbacks=[cb_saveWeights]
                            )

        print(str(i_CAE) + ' CAE trained', flush=True)
        # model.save_weights(weights_filepath)

        CAE_list[i_CAE]['weights_filename'] = weights_filepath
        CAE_list[i_CAE]['train_flag'] = False

        K.clear_session()
        # metrics = model.evaluate_generator(generator=generator_val, val_samples=int(np.ceil(len(subject_list_validation)/parameters[p.BATCH_SIZE])))
        # print('')
        # print('Metrics test: ')
        # print(metrics)
        # print('')

    print('Finished training')