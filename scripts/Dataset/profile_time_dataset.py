import numpy as np

import params as p
from params import params_VNET
from src.data_loader import Loader
from src.dataset import Dataset_train


def obtain_next(generator):
    return generator.__next__()

def obtain_next_times(generator, ntimes = 1):
    for i in range(ntimes):
        features, labels = obtain_next(generator)
        del features,labels


@profile
def test_memory_generator(generator, num_iters = 10 ):
    print(num_iters)
    for it in range(num_iters):
        print('\rIteration', it, '/', num_iters, flush=True)
        try:
            features, labels = obtain_next(generator)
            # del features, labels
        except StopIteration:
            break
    print('')

@profile
def test_memory_trainVal(generator_train, generator_val, n_epoch=1, n_iters_epoch = 10, n_iters_val = 10):
    print(n_epoch)
    for it_epoch in range(n_epoch):
        print('\rEpoch',it_epoch+1, '/', n_epoch, flush=True)
        try:
            features, labels = 0, 0
            for it_batch in range(n_iters_epoch):
                print('\r----- Iteration', it_batch+1, '/', n_iters_epoch, flush=True)
                features, labels = obtain_next(generator_train)

            obtain_next_times(generator_val,ntimes=n_iters_val)
            print('Validation performed')
            del features, labels

        except StopIteration:
            break
    print('')




if __name__ == '__main__':


    """ DATA LOADING """
    parameters = params_VNET.get_params()
    num_modalities = int(parameters[p.BOOLEAN_FLAIR]) + int(parameters[p.BOOLEAN_T1]) + int(parameters[p.BOOLEAN_T1C]) + int(parameters[p.BOOLEAN_T2])

    print('Loading data ...', flush=True)
    adni_db = Loader.create(parameters[p.DATABASE])
    subject_list = adni_db.load_subjects()

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

    print('TrainVal Dataset initialized', flush=True)
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)
    generator_train = dataset.data_generator(subject_list_train)
    generator_val = dataset.data_generator(subject_list_validation)

    test_memory_trainVal(generator_train,
                         generator_val,
                         n_epoch=2,
                         n_iters_epoch=int(np.ceil(1.0*len(subject_list_train)/parameters[p.BATCH_SIZE])),
                         n_iters_val=int(np.ceil(1.0*len(subject_list_validation)/parameters[p.BATCH_SIZE]))
                         )