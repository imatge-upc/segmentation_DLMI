import nibabel as nib
import numpy as np
from src.models import BratsModels
from src.dataset import BRATS_dataset_test
import params as p
from os.path import join
from src.helpers import io_utils
import argparse
import sys
from params import params_test as model_params
import SimpleITK as sitk

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

    parser.add_argument("-weights", help="Subject list: train or val", type=str)
    parser.add_argument("-listit", help="List iterator", type=int)
    arg = parser.parse_args(sys.argv[1:])
    model = arg.model
    filename = arg.filename
    n_subjects = arg.nsubjects
    filename_weights = arg.weights
    list_it = arg.listit



    params = model_params.get_params()
    params[p.MODEL_NAME] = model
    filepath = join(params[p.OUTPUT_PATH], 'test', filename + str(list_it) + '.txt')
    filepath_weights = join(params[p.OUTPUT_PATH], 'model_weights', filename_weights + '.h5')

    num_modalities = int(params[p.BOOLEAN_FLAIR]) + int(params[p.BOOLEAN_T1]) + int(params[p.BOOLEAN_T1C]) + \
                     int(params[p.BOOLEAN_T2])

    """ REDIRECT STDOUT TO FILE """
    io_utils.redirect_stdout_to_file(filepath=filepath)

    """ ARCHITECTURE DEFINITION """
    model, output_shape = BratsModels.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=params[p.MODEL_NAME],
        weights_filename=filepath_weights
    )

    model.summary()

    """ DATA LOADING """
    dataset = BRATS_dataset_test(input_dim=params[p.INPUT_DIM],
                                 output_shape=output_shape,
                                 data_path=params[p.DATA_PATH],
                                 n_classes=params[p.N_CLASSES],

                                 n_subjects=params[p.N_SUBJECTS],

                                 booleanFLAIR=params[p.BOOLEAN_FLAIR],
                                 booleanT1=params[p.BOOLEAN_T1],
                                 booleanT1c=params[p.BOOLEAN_T1C],
                                 booleanT2=params[p.BOOLEAN_T2],
                                 )


    affine = dataset.get_affine()

    """ MODEL TESTING """
    print
    print 'Testing started...'
    print 'Output_shape: ' + str(output_shape)

    subj_list = dataset.create_subjects()
    print(len(subj_list))

    global_metrics = 0
    if n_subjects == -1:
        n_subjects = len(subj_list)

    generator = dataset.data_generator(subject_list=subj_list[(list_it-1)*n_subjects:list_it*n_subjects])
    print 'Evaluate prediction'
    print
    print metric_list
    for subject in generator:
        image_modalities = subject[0]
        subject_id = subject[2]
        print subject_id

        print 'Predict classes...'
        predicted_subject = model.predict(image_modalities, batch_size=1, verbose=0)

        # Store predictions, and resize to have the same size as before
        print 'Storing predictions...'
        niftiImage = nib.Nifti1Image(np.squeeze(np.argmax(predicted_subject, axis=1)), affine)
        nib.save(niftiImage, join(params[p.OUTPUT_PATH], 'test', filename + subject_id + '_.nii.gz'))



    print 'Finished testing'

