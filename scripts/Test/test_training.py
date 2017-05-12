import nibabel as nib
import numpy as np
from src.models import BratsModels
from src.dataset import Dataset_train
import params as p
from os.path import join
from src.helpers import io_utils
import argparse
import sys
from params import params_train as model_params

metric_list = [
    'loss',
    'accuracy',
    'dice_whole',
    'dice_core',
    'dice_enhance',
    'recall_0',
    'recall_1',
    'recall_2',
    'recall_3',
    'recall_4',
    'precision_0',
    'precision_1',
    'precision_2',
    'precision_3',
    'precision_4',
]

if __name__ == "__main__":


    """ PARAMETERS """


    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Model", type=str)
    parser.add_argument("-filename", help="Filename", type=str)
    parser.add_argument("-nsubjects", help="Number of subjects", type=int)
    parser.add_argument("-slist", help="Subject list: train or val", type=str)
    parser.add_argument("-store", help="Subject list: train or val", type=int)
    parser.add_argument("-weights", help="Subject list: train or val", type=str)

    arg = parser.parse_args(sys.argv[1:])
    model = arg.model
    filename = arg.filename
    n_subjects = arg.nsubjects
    train_val = arg.slist
    store = arg.store
    filename_weights = arg.weights

    if store:
        print "Store=True"
    else:
        print "Store=False"

    params = model_params.get_params()
    filepath = join(params[p.OUTPUT_PATH], 'predictions', filename +'.txt')
    filepath_weights = join(params[p.OUTPUT_PATH], 'model_weights', filename_weights + '.h5')

    num_modalities = int(params[p.BOOLEAN_FLAIR]) + int(params[p.BOOLEAN_T1]) + int(params[p.BOOLEAN_T1C]) + \
                     int(params[p.BOOLEAN_T2])

    io_utils.redirect_stdout_to_file(filepath=filepath)


    """ ARCHITECTURE DEFINITION """

    model, output_shape = BratsModels.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=model,
        weights_filename=filepath_weights
    )

    """ REDIRECT STDOUT TO FILE """

    model.summary()

    """ DATA LOADING """
    dataset = Dataset_train(input_shape=params[p.INPUT_DIM],
                            output_shape=output_shape,
                            data_path=params[p.DATA_PATH],
                            n_classes=params[p.N_CLASSES],
                            n_subepochs=params[p.N_SUBEPOCHS],

                            sampling_scheme= params[p.SAMPLING_SCHEME],
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


    """ MODEL TESTING """
    print
    print 'Testing started...'
    print 'Output_shape: ' + str(output_shape)

    global_metrics = 0

    print 'Evaluate prediction'
    print
    print metric_list
    for subject in dataset.data_generator_inference(train_val=train_val):
        image_modalities = subject[0]
        labels=subject[1]
        subject_id = subject[2]
        print subject_id

        evaluation_metrics = model.evaluate(image_modalities, labels, batch_size=1, verbose=1, sample_weight=None)
        global_metrics += np.asarray(evaluation_metrics)
        print evaluation_metrics

        if store:
            print 'Predict classes...'
            predicted_subject = model.predict(image_modalities, batch_size=1, verbose=0)

            # Store predictions, and resize to have the same size as before
            print 'Storing predictions...'
            shape = dataset.get_subject(subject_id).get_subject_shape
            niftiImage = nib.Nifti1Image(np.squeeze(np.argmax(predicted_subject[:,:,:shape[0],:shape[1],:shape[2]],
                                                              axis=1)), dataset.get_affine())
            nib.save(niftiImage, join(params[p.OUTPUT_PATH], 'predictions', filename + subject_id + '_.nii.gz'))



    print
    print
    print 'Global metrics: '
    for index, metric in enumerate(global_metrics):
        print metric_list[index] + ': ' +str(metric/n_subjects)
    print 'Finished testing'

