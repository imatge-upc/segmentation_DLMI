import argparse
import sys

import numpy as np
from os.path import join

import params as p
from src.data_loader import Loader
from src.dataset import Dataset_train
from src.utils import io

if __name__ == "__main__":

    """ PARAMETERS """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    parameters = p.PARAMS_DICT[params_string].get_params()
    filepath = join(parameters[p.OUTPUT_PATH], 'logs', 'class_distributions' + '.txt')
    num_modalities = int(parameters[p.BOOLEAN_FLAIR]) + int(parameters[p.BOOLEAN_T1]) + int(parameters[p.BOOLEAN_T1C]) + \
                     int(parameters[p.BOOLEAN_T2])

    """ REDIRECT STDOUT TO FILE """
    io.redirect_stdout_to_file(filepath=filepath)

    """ DATA LOADING """
    brats_db = Loader.create(parameters[p.DATABASE])
    subject_list = brats_db.load_subjects()

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
                            num_modalities=num_modalities,

                            data_augmentation_flag=parameters[p.DATA_AUGMENTATION_FLAG],
                            class_weights=parameters[p.CLASS_WEIGHTS]
                            )



    print("Creating Generators...")
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)
    generator_train = dataset.data_generator(subject_list_train)
    generator_val = dataset.data_generator(subject_list_validation)


    """ CLASS DISTRIBUTION """

    print("Initial Distribution")
    class_freq = dataset.get_class_distribution(subject_list_train)
    print(class_freq)
    print()
    print()

    print('Training:')

    Ndistribution = np.zeros(parameters[p.N_CLASSES])
    Nsegments = 0
    for data in generator_train:
        Ndistribution = Ndistribution + 1.0 * np.bincount(np.argmax(data[1],axis=1).reshape(-1, ).astype(int),minlength = parameters[p.N_CLASSES])
        Nsegments = Nsegments + data[0].shape[0]
        if Nsegments >= parameters[p.N_SEGMENTS_VALIDATION]:
            break

    print(Ndistribution)
    print()
    print()


    print('Validation:')

    Ndistribution = np.zeros(parameters[p.N_CLASSES])
    Nsegments = 0
    for data in generator_val:

        Ndistribution = Ndistribution + 1.0 * np.bincount(np.argmax(data[1], axis=1).reshape(-1, ).astype(int),
                                                          minlength=parameters[p.N_CLASSES])
        Nsegments = Nsegments + data[0].shape[0]
        if Nsegments >= parameters[p.N_SEGMENTS_VALIDATION]:
            break


    print(Ndistribution)
    print()
    print()
    print()


