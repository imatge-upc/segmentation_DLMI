import sys
import numpy as np
import params as p
from params import params_train, params_pretrain,params_test,params_train_dense
from src.models import BratsModels
import argparse



if __name__ == "__main__":

    """ PARAMETERS """

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="Model", type=str)
    parser.add_argument('-p', help='Name of the model to be used', choices=p.PARAMS_DICT.keys())

    arg = parser.parse_args(sys.argv[1:])
    params_string = arg.p
    model = arg.m





    print 'Getting parameters to train the model...'
    params = p.PARAMS_DICT[params_string].get_params()
    num_modalities = int(params[p.BOOLEAN_FLAIR]) + int(params[p.BOOLEAN_T1]) + int(params[p.BOOLEAN_T1C]) + \
                     int(params[p.BOOLEAN_T2])


    """ ARCHITECTURE DEFINITION """
    model, output_shape = BratsModels.get_model(
        num_modalities=num_modalities,
        segment_dimensions=tuple(params[p.INPUT_DIM]),
        num_classes=params[p.N_CLASSES],
        model_name=model,
    )
    n_neurons = np.prod(params[p.INPUT_DIM])
    for layer in model.layers:
        n_neurons += np.prod(layer.output_shape[1:])

    print str(1.0*n_neurons*4/1000000) + " MB per image"







