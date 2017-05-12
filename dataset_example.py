
from os.path import join
import params as p
from params import params_example
from src.dataset import BRATS_dataset_semantic
import sys
import argparse
from params import params_pretrain as model_params
import numpy as np


if __name__ == "__main__":

    """ PARAMETERS """

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="Model", type=str)
    arg = parser.parse_args(sys.argv[1:])
    model = arg.m


    print 'Getting parameters to train the model...'
    params = model_params.get_params()
    params[p.MODEL_NAME] = model


    """ DATA LOADING """
    dataset = BRATS_dataset_semantic(input_dim=tuple(params[p.INPUT_DIM]),
                                     output_shape=tuple(params[p.INPUT_DIM]),
                                     data_path=params[p.DATA_PATH],
                                     n_classes=params[p.N_CLASSES],
                                     n_subepochs=params[p.N_SUBEPOCHS],

                                     probabilities_train=params[p.PROB_TRAIN],
                                     n_subjects_per_epoch_train=params[p.N_SUBJECTS_TRAIN],
                                     n_segments_per_epoch_train=params[p.N_SEGMENTS_TRAIN],

                                     probabilities_validation=params[p.PROB_VALIDATION],
                                     n_subjects_per_epoch_validation=params[p.N_SUBJECTS_VALIDATION],
                                     n_segments_per_epoch_validation=params[p.N_SEGMENTS_VALIDATION],

                                     id_list_train=params[p.ID_LIST_TRAIN],
                                     id_list_validation=params[p.ID_LIST_VALIDATION],
                                     booleanFLAIR=params[p.BOOLEAN_FLAIR],
                                     booleanT1=params[p.BOOLEAN_T1],
                                     booleanT1c=params[p.BOOLEAN_T1C],
                                     booleanT2=params[p.BOOLEAN_T2],
                                     class_weights=params[p.CLASS_WEIGHTS]
                                     )

    # generator_train = dataset.data_generator(batch_size=params[p.BATCH_SIZE], train_val='train')
    print "Creating generator for validation"
    generator_val = dataset.data_generator(batch_size=params[p.BATCH_SIZE], train_val='val')

    num_iter = int(np.ceil(params[p.N_SEGMENTS_VALIDATION]/params[p.BATCH_SIZE])*10)
    print "Start iteration"
    for i in range(num_iter):
        data = generator_val.next()
        print i
