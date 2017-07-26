import params as p
from params import params_VNET as pVNET
from src.data_loader import Loader
from src.dataset import Dataset_train

if __name__ == "__main__":

    """ PARAMETERS """
    print('Getting parameters to train the model...',flush=True)
    parameters = pVNET.get_params()


    """ DATA LOADING """
    print('Loading dataset ...',flush=True)
    brats_db = Loader.create(parameters[p.DATABASE])
    subject_list = brats_db.load_subjects()

    print("Number of subjects loaded: " + str(len(subject_list)),flush=True)

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
                            num_modalities=4,

                            data_augmentation_flag=parameters[p.DATA_AUGMENTATION_FLAG],
                            class_weights=parameters[p.CLASS_WEIGHTS]
                            )


    print('TrainVal Dataset initialized',flush=True)
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)
    generator_train = dataset.data_generator(subject_list_train)
    generator_val = dataset.data_generator(subject_list_validation)

    for data in generator_train:
        print('New batch: ')
        assert data[0].shape == (parameters[p.BATCH_SIZE],) + parameters[p.INPUT_DIM] + (4,)
        assert data[1].shape == (parameters[p.BATCH_SIZE],) + parameters[p.INPUT_DIM] + (4,)
        print('------- correct shape OK!')
        break