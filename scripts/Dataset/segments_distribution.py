from os.path import join
import numpy as np
import params as p
from src.dataset import Dataset_train
from src.helpers import io_utils
import argparse
import sys

if __name__ == "__main__":

    arguments_parser = argparse.ArgumentParser()
    arguments_parser.add_argument('-params', help='Name of the model to be used', choices=p.PARAMS_DICT.keys())
    args = arguments_parser.parse_args(sys.argv[1:])
    params_string = args.params


    """ PARAMETERS """
    print "Defining parameters..."
    params = p.PARAMS_DICT[params_string].get_params()
    filepath = join(params[p.OUTPUT_PATH], 'logs', 'class_distributions' + '.txt')
    num_modalities = int(params[p.BOOLEAN_FLAIR]) + int(params[p.BOOLEAN_T1]) + int(params[p.BOOLEAN_T1C]) + \
                     int(params[p.BOOLEAN_T2])

    """ REDIRECT STDOUT TO FILE """
    io_utils.redirect_stdout_to_file(filepath=filepath)


    """ DATA LOADING """
    print "Dataset initialization..."
    dataset = Dataset_train(input_shape=params[p.INPUT_DIM],
                            output_shape=params[p.INPUT_DIM],
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

                            class_weights=params[p.CLASS_WEIGHTS]
                            )



    print "Creating Generators..."
    generator_train = dataset.data_generator(batch_size=params[p.BATCH_SIZE], train_val='train')
    generator_val = dataset.data_generator(batch_size=params[p.BATCH_SIZE], train_val='val')


    """ CLASS DISTRIBUTION """

    print "Initial Distribution"
    class_freq = dataset.get_class_distribution()
    print class_freq
    print
    print

    print 'Training:'

    Ndistribution = np.zeros(params[p.N_CLASSES])
    Nsegments = 0
    for data in generator_train:
        Ndistribution = Ndistribution + 1.0 * np.bincount(np.argmax(data[1],axis=1).reshape(-1, ).astype(int),minlength = params[p.N_CLASSES])
        Nsegments = Nsegments + data[0].shape[0]
        if Nsegments >= params[p.N_SEGMENTS_VALIDATION]:
            break

    print Ndistribution
    print
    print
    print


    print 'Validation:'

    Ndistribution = np.zeros(params[p.N_CLASSES])
    Nsegments = 0
    for data in generator_val:

        Ndistribution = Ndistribution + 1.0 * np.bincount(np.argmax(data[1], axis=1).reshape(-1, ).astype(int),
                                                          minlength=params[p.N_CLASSES])
        Nsegments = Nsegments + data[0].shape[0]
        if Nsegments >= params[p.N_SEGMENTS_VALIDATION]:
            break


    print  Ndistribution
    print
    print
    print
