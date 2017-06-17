import argparse
import sys
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from os.path import join
from src.callbacks import LearningRateDecayAccuracyPlateaus
from src.dataset import Dataset_train
from src.helpers import io_utils
from src.models import BratsModels
from keras import backend as K
from keras.utils import plot_model
import numpy as np
from src.callbacks import exponential_decay


import params as p

if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Filename", type=str)
    parser.add_argument("-m", help="Model", type=str)
    parser.add_argument("-w", help="Weights", default=None, type=str)
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    filename = arg.f
    load_weights_file = arg.w
    model = arg.m
    params_string = arg.p


    print 'Getting parameters to train the model...'
    params = p.PARAMS_DICT[params_string].get_params()
    filepath = join(params[p.OUTPUT_PATH], 'logs', filename + '.txt')
    filepath_weights = join(params[p.OUTPUT_PATH], 'model_weights', filename + '.h5')
    load_weights_filepath = join(params[p.OUTPUT_PATH], 'model_weights', load_weights_file + '.h5') if load_weights_file != -1 else None

    num_modalities = int(params[p.BOOLEAN_FLAIR]) + int(params[p.BOOLEAN_T1]) + int(params[p.BOOLEAN_T1C]) + int(params[p.BOOLEAN_T2])



    """ REDIRECT STDOUT TO FILE """
    print 'Output redirected to file... '
    print 'Suggestion: Use tail command to see the output'
    io_utils.redirect_stdout_to_file(filepath=filepath)




    """ ARCHITECTURE DEFINITION """
    model, output_shape = BratsModels.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name='u_net',
        weights_filename=load_weights_filepath
    )


    model.summary()
    plot_file = join(params[p.OUTPUT_PATH], filename +'.png')
    plot_model(model, to_file=plot_file)



    print "Architecture defined ..."
    """ DATA LOADING """
    dataset = Dataset_train(input_shape=tuple(params[p.INPUT_DIM]),
                            output_shape=tuple(params[p.INPUT_DIM]),
                            data_path=params[p.DATA_PATH],
                            n_classes=params[p.N_CLASSES],
                            n_subepochs=params[p.N_SUBEPOCHS],

                            sampling_scheme=params[p.SAMPLING_SCHEME],
                            sampling_weights=params[p.SAMPLING_WEIGHTS],

                            n_subjects_per_epoch_train=params[p.N_SUBJECTS_TRAIN],
                            n_segments_per_epoch_train=params[p.N_SEGMENTS_TRAIN],

                            n_subjects_per_epoch_validation=params[p.N_SUBJECTS_VALIDATION],
                            n_segments_per_epoch_validation=params[p.N_SEGMENTS_VALIDATION],

                            id_list_train=params[p.ID_LIST_TRAIN],
                            id_list_validation=params[p.ID_LIST_VALIDATION],
                            booleanFLAIR=params[p.BOOLEAN_FLAIR],
                            booleanT1=params[p.BOOLEAN_T1],
                            booleanT1c=params[p.BOOLEAN_T1C],
                            booleanT2=params[p.BOOLEAN_T2],
                            booleanROImask=params[p.BOOLEAN_ROImask],
                            booleanLabels=params[p.BOOLEAN_LABELS],

                            data_augmentation_flag= params[p.DATA_AUGMENTATION_FLAG],
                            class_weights=params[p.CLASS_WEIGHTS]
                            )

    print "Brats Dataset initialized"
    generator_train = dataset.data_generator(batch_size=params[p.BATCH_SIZE], train_val='train')
    generator_val = dataset.data_generator(batch_size=params[p.BATCH_SIZE], train_val='val')

    num_epochs = 35
    cb_saveWeights = ModelCheckpoint(filepath=filepath_weights, save_best_only=True, mode='min')
    cb_earlyStopping = EarlyStopping(patience=5)
    cb_learningRateScheduler = LearningRateDecayAccuracyPlateaus(decay_rate=2)
    cb_learningRateScheduler2 = ReduceLROnPlateau(factor=0.5, patience=2)
    #cb_learningRateScheduler = LearningRateScheduler(schedule=exponential_decay(initial_lrate=0.0005, decay=0.5, epochs=num_epochs))

    """ MODEL TRAINING """
    print
    print 'Training started...'
    #print 'Samples per epoch: ' + str(params[p.N_SEGMENTS_TRAIN])
    print 'Steps per epoch: ' + str(params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE])
    print 'Output_shape: ' + str(output_shape)

    model.fit_generator(generator=generator_train,
                        steps_per_epoch=int(np.ceil(params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE])), #posar el nombre d'iteracions=segments/batchsize (160 i no 800)
                        epochs=num_epochs,
                        validation_data=generator_val,
                        validation_steps=int(np.ceil(params[p.N_SEGMENTS_VALIDATION]/params[p.BATCH_SIZE])),
                        callbacks=[cb_saveWeights],#, cb_learningRateScheduler2],
                        class_weight=dataset.class_weights)

    print 'Finished training'
