import nibabel as nib
import numpy as np
from src.models import BratsModels
from src.dataset import Dataset_train
import params as p
from os.path import join
from src.helpers import io_utils, visualization
import argparse
import sys


if __name__ == "__main__":

    """ PARAMETERS """
    print "Checking parameters ..."
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Model", type=str)
    parser.add_argument("-layer", help="Layer name", type=str)
    parser.add_argument("-weights", help="Subject list: train or val", type=str)
    parser.add_argument("-params", help="Parameters type string", type=str)

    arg = parser.parse_args(sys.argv[1:])
    model = arg.model
    filename_weights = arg.weights
    layer = arg.layer
    params_string = arg.params




    params = p.PARAMS_DICT[params_string].get_params()
    filepath_weights = join(params[p.OUTPUT_PATH], 'model_weights', filename_weights + '.h5')

    num_modalities = int(params[p.BOOLEAN_FLAIR]) + int(params[p.BOOLEAN_T1]) + int(params[p.BOOLEAN_T1C]) + \
                     int(params[p.BOOLEAN_T2])

    """ ARCHITECTURE DEFINITION """
    model, output_shape = BratsModels.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=model,
        weights_filename=filepath_weights
    )

    # model.summary()

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


    affine = dataset.get_affine()
    subject_id ='LGG_brats_2013_pat0003_1'

    """ MODEL TESTING """
    print
    print 'subject:' + subject_id
    print 'Output_shape: ' + str(output_shape)




    print
    print 'Getting activations ...'
    print
    subject = dataset.get_subject(subject_id)
    image_modalities, image_labels, _ = dataset.data_generator_one(subject)

    print 'Predict classes...'
    predicted_subject = np.squeeze(visualization.get_activations(model,layer,image_modalities))

    print 'Lenght of this layer'
    print predicted_subject.shape

    # Store predictions, and resize to have the same size as before
    print 'Storing predictions...'
    for it_image,image in enumerate(predicted_subject):
        niftiImage = nib.Nifti1Image(image[:], affine)
        nib.save(niftiImage, join(params[p.OUTPUT_PATH], 'activations', model +'_layer_'+str(layer)+'_' + subject_id +'_' + str(it_image) + '_.nii.gz'))



    print 'Finished testing'

