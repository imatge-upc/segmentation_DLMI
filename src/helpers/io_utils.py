import csv
import sys
import numpy as np
import os
import keras.backend as K

def redirect_stdout_to_file(filepath):
    """
    Redirects the standard output (STDOUT) to the file specified by filepath

    Parameters
    ----------
    filepath : String
       Path to the file where all standard output should be redirected
    """
    try:
        sys.stdout = open(filepath, 'wb')
    except IOError:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path)
        sys.stdout = open(filepath, 'wb')


def get_loss_lines(filename, metric, metric_dictionary):
    """
    Returns the training and validation values for the specified metric

    Parameters
    ----------
    filename : String
        Path to the log file where all data should be extracted
    metric : String
        Name of the metric to be extracted
    metric_dictionary : Dictionary
        Dictionary that maps the name of the metric with the index where it is found in the log file

    Returns
    -------
    tuple
        Tuple with two arrays, one containing the training metrics and another containing the validation ones
    """
    train_l = []
    val_l = []

    with open(filename, 'rU') as csvfile:

        lines = csv.reader(csvfile, delimiter='\n')

        for line in lines:

            if len(line) > 0:

                line = line[0]

                if metric in line:
                    spline = line.split(' - ')

                    if 'val_' + metric in line:

                        try:
                            vloss = spline[metric_dictionary[metric] + len(metric_dictionary)].split('val_' + metric + ': ')[1]
                            vloss = float(vloss.split('\b')[0])
                        except:
                            # print spline
                            vloss = float('nan')

                        val_l.append(vloss)

                        tloss = spline[metric_dictionary[metric]].split(metric + ': ')[1]
                        tloss = float(tloss.split(' ')[0])

                    else:
                        try:
                            # quickfix for some lines in the log file
                            # that don't contain the loss value for some reason
                            tloss = spline[metric_dictionary[metric]].split(metric + ': ')[1]
                            tloss = float(tloss.split('\b')[0])
                        except:
                            tloss = float('nan')

                    train_l.append(tloss)

    return train_l, val_l

def compute_class_frequencies(segment,num_classes):
    if isinstance(segment,list):
        segment = np.asarray(segment)
    f = 1.0 * np.bincount(segment.reshape(-1,).astype(int),minlength=num_classes) / np.prod(segment.shape)
    return f

def compute_centralvoxel_frequencies(segment,minlength):
    if isinstance(segment,list):
        segment = np.asarray(segment)
    shape = segment.shape[-3:]

    middle_coordinate = np.zeros(3,int)
    for it_coordinate,coordinate in enumerate(shape):
        if coordinate%2==0:
            middle_coordinate[it_coordinate] = coordinate / 2 - 1
        else:
            middle_coordinate[it_coordinate] = coordinate/2

    segment = segment.reshape((-1,) + shape)
    f = 1.0 * np.bincount(segment[:,middle_coordinate[0],middle_coordinate[1],middle_coordinate[2]].reshape(-1,).astype(int),minlength=minlength) / np.prod(segment.shape[:-3])
    return f

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations