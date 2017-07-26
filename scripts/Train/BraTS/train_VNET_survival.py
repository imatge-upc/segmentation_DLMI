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
from src.callbacks import LearningRateExponentialDecay

if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = p.PARAMS_DICT[params_string].get_params()
    filename = params[p.MODEL_NAME]
    dir_path = join(params[p.OUTPUT_PATH], 'LR_' + str(params[p.LR]) + '_DA_')

    logs_filepath = join(dir_path, 'logs', filename + '.txt')
    weights_filepath = join(dir_path, 'model_weights', filename + '.h5')
    weights_initialize = join('/work/acasamitjana/segmentation/BraTS/brats2017/survival', '05_lrd.h5')

    """ REDIRECT STDOUT TO FILE """
    print('Output redirected to file... ')
    print('Suggestion: Use tail command to see the output')
    io.create_results_dir(dir_path=dir_path)
    # io.redirect_stdout_to_file(filepath=logs_filepath)

    print('PARAMETERS')
    print(params)
    print('Learning rate exponential decay')


    """ ARCHITECTURE DEFINITION """
    premodel, output_shape = BraTS_models.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        l1=0.000,
        l2=0.00

    )
    premodel.load_weights(weights_initialize)
    # model = BraTS_models.compile(model, lr=params[p.LR], optimizer_name='Adam',model_name='survival')

    model, output_shape = BraTS_models.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name='survival_net',
        premodel=premodel,
        l1=0.0001,
        l2=0.001
    )
    model = BraTS_models.compile(model, lr=params[p.LR], optimizer_name='Adam',model_name='survival')

    model.summary()
    plot_file = join(dir_path, filename +'.png')
    plot_model(model, to_file=plot_file)

    json_string = model.to_json()
    open(join(dir_path, filename + '_model.json'), 'w').write(json_string)


    print("Architecture defined ...")
    """ DATA LOADING """
    brats_db = Loader.create(config_dict=DB.BRATS2017)
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

    print('TrainVal Dataset initialized')
    subject_list = list(filter(lambda subject: subject.load_survival() is not None, subject_list ))
    survival_mean = 0
    survival_var= 0
    n_sbj = 0
    for subject in subject_list:
        survival = subject.load_survival()
        if survival is not None:
            survival_mean += survival
            n_sbj += 1
    survival_mean = survival_mean/n_sbj

    for subject in subject_list:
        survival = subject.load_survival()
        if survival is not None:
            survival_var += (survival - survival_mean)**2
    survival_var = survival_var/n_sbj

    print(survival_mean)
    print(survival_var)
    for subject in subject_list:
        subject.survival_mean = survival_mean
        subject.survival_std = np.sqrt(survival_var)

    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)
    class_weights = np.asarray([1.,13.43895481])#dataset.class_weights(subject_list)


    steps_per_epoch = int(len(subject_list_train)) if params[p.DATA_AUGMENTATION_FLAG] is False else int(2*len(subject_list_train))
    validation_steps = int(len(subject_list_validation))
    generator_train = dataset.data_generator_BraTS_survival(subject_list_train, mode='train')
    generator_val = dataset.data_generator_BraTS_survival(subject_list_validation, mode='validation')


    cb_saveWeights = ModelCheckpoint(filepath=weights_filepath, save_weights_only=True)
    cb_earlyStopping = EarlyStopping(patience=5)
    cb_learningRateScheduler = LearningRateExponentialDecay(epoch_n=0,num_epoch=params[p.N_EPOCHS])
    callbacks_list = [cb_saveWeights]


    """ MODEL TRAINING """
    print()
    print('Training started...')
    print('Steps per epoch: ' + str(steps_per_epoch))
    print('Output_shape: ' + str(output_shape))
    print('Class weight' + str(class_weights))



    model.fit_generator(generator=generator_train,
                        steps_per_epoch=steps_per_epoch,
                        epochs=params[p.N_EPOCHS],
                        validation_data=generator_val,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list,
                        )#class_weight=class_weights)

    print('Finished training')
