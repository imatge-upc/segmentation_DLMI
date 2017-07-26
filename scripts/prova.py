import argparse
import sys
import os
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from os.path import join
from database.WMH.data_loader import Loader
from src.dataset import Dataset_train
from src.utils import io
from src.models import SegmentationModels, WMH_models
from src.config import DB
from src.callbacks import LearningRateExponentialDecay, LearningRateDecay
from keras.utils import plot_model
import numpy as np


import params.WMH as p

if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    #
    # params[p.BATCH_SIZE] = 1
    # params[p.INPUT_DIM] = (32,32,32)
    filename = params[p.MODEL_NAME] + '_continue'
    dir_path = join(params[p.OUTPUT_PATH], 'LR_' + str(params[p.LR])+'_DA_6_4_32size')

    logs_filepath = join(dir_path, 'logs', filename + '.txt')
    load_weights_filepath = join(dir_path, 'model_weights', filename + '.h5')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')

    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    # io.create_results_dir(dir_path=dir_path)
    # io.redirect_stdout_to_file(filepath=logs_filepath)

    print('PARAMETERS')
    print(params)
    print('ADDING CLASS WEIGHTS')
    print('Learning rate exponential decay')



    print("Architecture defined ...")
    """ DATA LOADING """
    whm_db = Loader.create(config_dict=DB.WMH)
    subject_list = whm_db.load_subjects()

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

    print('TrainVal Dataset initialized')
    class_weights = dataset.class_weights(subject_list)
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)

    steps_per_epoch = 6 if params[p.DATA_AUGMENTATION_FLAG] is False else int(2*len(subject_list_train))
    validation_steps = 4



    generator_train = dataset.data_generator_AE(subject_list_train)
    generator_val = dataset.data_generator_AE(subject_list_validation)



    cb_saveWeights = ModelCheckpoint(filepath=weights_filepath)
    cb_earlyStopping = EarlyStopping(patience=5)
    cb_learningRateScheduler = LearningRateDecay(epoch_n=0,decay=0.001)
    callbacks_list = [cb_saveWeights,cb_learningRateScheduler]


    """ MODEL TRAINING """

    """ ARCHITECTURE DEFINITION """

    from keras.models import Sequential, Model
    from keras.layers import Conv3D, Activation, Input, Lambda, BatchNormalization
    from src.layers import BatchNormalizationMasked, repeat_channels_shape, repeat_channels

    x = Input( shape = tuple(params[p.INPUT_DIM])+(2,))
    mask = Input(shape = tuple(params[p.INPUT_DIM])+(1,))
    mask_0 = Lambda(repeat_channels(8), output_shape=repeat_channels_shape(8),
                    name='repeat_channel_input')(mask)
    tmp = Conv3D(8,(3,3,3),padding='same')(x)
    tmp = BatchNormalizationMasked(axis=4)([tmp,mask_0])
    # tmp = BatchNormalization(axis=4)(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv3D(2,(3,3,3),padding='same')(tmp)

    model = Model(inputs=[x, mask], outputs = [tmp])

    model.compile('Adam',loss='mse')
    model.summary()
    plot_file = join(dir_path, filename + '.png')
    plot_model(model, to_file=plot_file)

    json_string = model.to_json()
    open(join(dir_path, filename + '_model.json'), 'w').write(json_string)

    print('')
    print('Training started...')
    print('Steps per epoch: ' + str(steps_per_epoch))
    print('Class weight' + str(class_weights))

    model.fit_generator(generator=generator_train,
                        steps_per_epoch=steps_per_epoch,
                        epochs=params[p.N_EPOCHS],
                        validation_data=generator_val,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list,
                        class_weight=class_weights)

    print('Finished training')
