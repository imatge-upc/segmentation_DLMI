import argparse
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from os.path import join

import params.BraTS as p
from src.config import DB
from database.BraTS.data_loader import Loader
from src.dataset import Dataset_Brats
from src.models import BraTS_models
from src.utils import io
from src.callbacks import LearningRateExponentialDecay, LearningRatePredefinedDecay

if __name__ == "__main__":

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

    print('PARAMETERS')
    print(params)

    """ ARCHITECTURE DEFINITION """
    model, output_shape = BraTS_models.get_model(
        num_modalities=params[p.NUM_MODALITIES],
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        l1=0.00001,
        l2=0.005,
        momentum=0.99,
    )
    model = BraTS_models.compile(model, lr=params[p.LR], loss_name=params[p.LOSS])


    model.summary()
    plot_file = join(dir_path, filename +'.png')
    plot_model(model, to_file=plot_file)

    json_string = model.to_json()
    open(join(dir_path, filename + '_model.json'), 'w').write(json_string)


    print("Architecture defined ...")
    """ DATA LOADING """
    brats_db = Loader.create(config_dict=DB.BRATS2017)
    subject_list = brats_db.load_subjects()

    dataset = Dataset_Brats(input_shape=tuple(params[p.INPUT_DIM]),
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

    print("BraTS Dataset initialized")
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)

    steps_per_epoch = int(len(subject_list_train)) if params[p.DATA_AUGMENTATION_FLAG] is False else int(len(params[p.DATA_AUGMENTATION_PLANES])*len(subject_list_train))
    validation_steps = int(len(subject_list_validation))
    generator_train = dataset.data_generator_full(subject_list_train, mode='train')
    generator_val = dataset.data_generator_full(subject_list_validation, mode='validation')



    cb_saveWeights = ModelCheckpoint(filepath=weights_filepath, save_best_only=True)
    cb_earlyStopping = EarlyStopping(patience=5)
    cb_learningRateScheduler = LearningRateExponentialDecay(epoch_n=0,num_epoch=params[p.N_EPOCHS])

    callbacks_list = [cb_saveWeights,cb_learningRateScheduler]


    """ MODEL TRAINING """
    print()
    print('Training started...')
    print('Steps per epoch: ' + str(steps_per_epoch))
    print('Output_shape: ' + str(output_shape))

    model.fit_generator(generator=generator_train,
                        steps_per_epoch=steps_per_epoch,
                        epochs=params[p.N_EPOCHS],
                        validation_data=generator_val,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list)

    print('Finished training')
