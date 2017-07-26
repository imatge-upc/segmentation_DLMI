import keras.backend as K
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
import params as p
from database.data_loader import Loader
from src.dataset import Dataset_train
from src.models import SegmentationModels, BraTS_models

plt.switch_backend('TKAgg')

def read_model_activations(file_path, layer_name, parameters):

    """ DATA LOADING """
    db = Loader.create(parameters[p.DATABASE])
    subject_list = db.load_subjects()


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
                            num_modalities=parameters[p.NUM_MODALITIES],

                            data_augmentation_flag=parameters[p.DATA_AUGMENTATION_FLAG],
                            class_weights=parameters[p.CLASS_WEIGHTS]
                            )


    print('TrainVal Dataset initialized')
    subject_list_train, subject_list_validation = dataset.split_train_val(subject_list)


    """ MODEL TESTING """
    # print('')
    # print('Testing started...')
    # print('Samples per epoch:' + str(len(subject_list_validation)))


    """ ARCHITECTURE DEFINITION """
    print('Defining architecture... ', flush=True)
    model, output_shape = BraTS_models.get_model(
        num_modalities=parameters[p.NUM_MODALITIES],
        segment_dimensions=tuple(parameters[p.INPUT_DIM]),
        num_classes=parameters[p.N_CLASSES],
        model_name=parameters[p.MODEL_NAME],
        # BN_last = parameters[p.BN_LAST],
        # shortcut_input = parameters[p.SHORTCUT_INPUT],
        # mask_bool = parameters[p.BOOL_MASK],
    )


    # model.load_weights(file_path)
    model = BraTS_models.compile(model, lr=parameters[p.LR],model_name='segmentation')

    # model.summary()
    print('Architeture defined: ')

    if parameters[p.BOOL_MASK]:
        generator_train = dataset.data_generator_BraTS_seg(subject_list_train, mode='train')
        get_output = K.function([model.get_layer('V-net_input').input, model.get_layer('input_1').input, K.learning_phase()],
                                [model.get_layer(layer_name).output])

        image, labels = next(generator_train)
        predicted_activations = get_output([image[0], image[2], True])[0]

    else:
        generator_train = dataset.data_generator_full(subject_list_train, mode='train')
        get_output = K.function([model.layers[0].input, K.learning_phase()],
                                [model.get_layer(layer_name).output])

        image, labels = next(generator_train)
        predicted_activations = get_output([image, True])[0]
        # predicted_activations = predicted_activations * image[1]

    HISTOGRAM = False
    FEATURE_MAPS = True
    if HISTOGRAM:

        plt.figure()
        plt.plot(predicted_activations.flatten())
        # plt.hist(predicted_activations.flatten(),bins = 100)
        plt.show()

    elif FEATURE_MAPS:
        # y_pred_decision = np.floor(predicted_activations / np.max(predicted_activations, axis=4, keepdims=True))
        #
        # mask_true = labels[:, :, :, :, 0]
        # mask_pred = y_pred_decision[:, :, :, :, 0]
        #
        # y_sum = np.sum(mask_true * mask_pred)
        #
        # print( (y_sum+ np.finfo(float).eps) / (np.sum(mask_true) + np.finfo(float).eps))
        # mask_true = labels[:, :, :, :, 1]
        # mask_pred = predicted_activations[:, :, :, :, 1]
        #
        # y_sum = np.sum(mask_true * mask_pred)
        #
        # print(- (2. * y_sum + np.finfo(float).eps) / (np.sum(mask_true) + np.sum(mask_pred) + np.finfo(float).eps))

        # predicted_activations = y_pred_decision
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(predicted_activations[0,:,:,100,0])
        plt.colorbar()

        plt.subplot(2,2,2)
        plt.imshow(predicted_activations[0,:,:,100,1])
        plt.colorbar()

        plt.subplot(2,2,3)
        plt.imshow(predicted_activations[0,:,:,100,2])
        plt.colorbar()

        plt.subplot(2,2,4)
        plt.imshow(predicted_activations[0,:, :, 100,3])
        plt.colorbar()
        plt.show()

    print('Finished testing')

if __name__ == "__main__":

    from params.BraTS import params_VNET_full as parameters

    layer_name = 'repeat_mask'
    dir_path = '/work/acasamitjana/segmentation/BraTS/brats2017/LR_0.005_DA_False_seg_only/model_weights'
    file = 'brats2017_seg_only.h5'
    w_file_path = join(dir_path, file)

    read_model_activations(w_file_path, layer_name, parameters.get_params())
