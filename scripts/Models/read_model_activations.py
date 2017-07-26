import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from os.path import join

import params as p
from database.data_loader import Loader
from src.dataset import Dataset_train
from src.models import SegmentationModels

plt.switch_backend('TKAgg')

def read_model_activations(file_path, layer_name, parameters):


    # """ REDIRECT STDOUT TO FILE """
    # print('Output redirected to file... ')
    # print('Suggestion: Use tail command to see the output')
    # logfile_path = join(dir_path,  'logs', parameters[p.MODEL] + '_results.txt')
    # io.redirect_stdout_to_file(filepath=logfile_path)


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
    model, output_shape = SegmentationModels.get_model(
        num_modalities=parameters[p.NUM_MODALITIES],
        segment_dimensions=tuple(parameters[p.INPUT_DIM]),
        num_classes=parameters[p.N_CLASSES],
        model_name=parameters[p.MODEL_NAME],
        BN_last = parameters[p.BN_LAST],
        shortcut_input = parameters[p.SHORTCUT_INPUT],
        mask_bool=parameters[p.BOOL_MASK],
    )


    # model.load_weights(file_path)
    model = SegmentationModels.compile(model, lr=parameters[p.LR],num_classes=parameters[p.N_CLASSES])

    # model.summary()
    print('Architeture defined')
    outputs = []
    layer_names = []
    for layer in model.layers:
        if layer.name == layer_name or layer_name is None:
            if isinstance(layer.output, list):
                for it_out, out in enumerate(layer.output):
                    layer_names.append(layer.name + '_' + str(it_out))
                    outputs.append(out)
            else:
                layer_names.append(layer.name)
                outputs.append(layer.output)


    if parameters[p.BOOL_MASK]:
        generator_train = dataset.data_generator_full_mask(subject_list_train, mode='train')
        funcs = [
            K.function([model.get_layer('V-net_input').input, model.get_layer('V-net_mask').input, K.learning_phase()],
                       [out]) for out in outputs
            ]

        image, labels = next(generator_train)
        layer_outputs = [func([image[0], image[1], False])[0] for func in funcs]

    else:
        generator_train = dataset.data_generator_full(subject_list_train, mode='train')
        funcs = [
            K.function([model.get_layer('V-net_input').input, K.learning_phase()],
                       [out]) for out in outputs
            ]

        image, labels = next(generator_train)
        layer_outputs = [func([image[0], False])[0] for func in funcs]









    for layer_n,layer_out in zip(layer_names,layer_outputs):

        print('Layer: ' + layer_n + ' Sum: ' + str(np.sum(layer_out)))


    print('Finished testing')

if __name__ == "__main__":
    # Parse arguments from command-line
    # arguments_parser = argparse.ArgumentParser()
    # arguments_parser.add_argument('-w', help='Name of weights file')
    # args = arguments_parser.parse_args(sys.argv[1:])
    #
    # w_file_path = args.w

    from params.BraTS import params_VNET_1 as parameters

    layer_name = 'Softmax'
    dir_path = '/work/acasamitjana/segmentation/BraTS/VNet_1/LR_0.0005_dice_DA_False_BN_False_shortcut_False_mask_True/model_weights'
    file = 'v_net.h5'
    w_file_path = join(dir_path, file)
    print(w_file_path)
    read_model_activations(w_file_path, layer_name, parameters.get_params())