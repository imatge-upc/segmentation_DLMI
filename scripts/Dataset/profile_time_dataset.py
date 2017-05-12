from src.dataset import Dataset_train
import params as p
from params import params_train as model_params
import numpy as np
import argparse
import sys

arguments_parser = argparse.ArgumentParser()
arguments_parser.add_argument('-params', help='Name of the model to be used', choices=p.PARAMS_DICT.keys())

args = arguments_parser.parse_args(sys.argv[1:])
params_string = args.params
params = p.PARAMS_DICT[params_string].get_params()


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
generator_train = dataset.data_generator(batch_size=params[p.BATCH_SIZE],train_val='train')
generator_val = dataset.data_generator(batch_size=1,train_val='val_full')

# @profile
def obtain_next(generator):
    return generator.next()

def obtain_next_times(generator, ntimes = 1):
    for i in range(ntimes):
        features, labels = obtain_next(generator)
        del features,labels

@profile
def test_memory_load_image_segments():
    image_segments, labels_segments = dataset.load_image_segments(dataset.create_subjects())
    print

@profile
def test_memory_generator(generator, num_iters = 10 ):
    print num_iters
    for it in xrange(num_iters):
        print '\rIteration', it, '/', num_iters,
        try:
            features, labels = obtain_next(generator)
            # del features, labels
        except StopIteration:
            break
    print

@profile
def test_memory_trainVal(generator_train,generator_val, num_iters_train = 10, num_iters_val = 10):
    print num_iters_train
    for it in xrange(num_iters_train):
        print '\rIteration', it, '/', num_iters_train,
        try:
            features, labels = obtain_next(generator_train)
            if (it+1)%int(np.ceil(params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE])) == 0:
                obtain_next_times(generator_val,ntimes=num_iters_val)
                print "Validation performed"
            del features, labels

        except StopIteration:
            break
    print




if __name__ == '__main__':

    test_memory_trainVal(generator_train,
                         generator_val,
                         num_iters_train = int(np.ceil(1.5*params[p.N_SEGMENTS_TRAIN]/params[p.BATCH_SIZE])),
                         num_iters_val=params[p.N_SUBJECTS_VALIDATION]
                         )
