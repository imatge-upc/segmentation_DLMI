import csv
import sys
import numpy as np
import os
from os.path import exists, join


def redirect_stdout_to_file(filepath):
    """
    Redirects the standard output (STDOUT) to the file specified by filepath

    Parameters
    ----------
    filepath : String
       Path to the file where all standard output should be redirected
    """
    try:
        sys.stdout = open(filepath, 'a')
    except IOError:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path)
        sys.stdout = open(filepath, 'a')



def get_loss_lines(filename, metric):
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
    iter_epoch = 0
    bool_iter_epoch = True
    metric = metric+': '
    with open(filename, 'rU') as csvfile:

        lines = csv.reader(csvfile, delimiter='\n')

        for line in lines:

            if len(line) > 0:

                line = line[0]

                if metric in line:
                    spline = line.split(' - ')

                    if 'val_' + metric in line:
                        if bool_iter_epoch:
                            iter_epoch += 1
                            bool_iter_epoch = False

                        try:
                            vloss = 0
                            for s in spline:
                                if 'val_' + metric in s:
                                    vloss = s.split('val_' + metric)[-1]
                                    break
                            if vloss == 0:
                                raise ValueError('This metric is not found for validation')
                            vloss = float(vloss.split('\b')[0])
                        except:
                            vloss = float('nan')

                        val_l.append(vloss)

                        tloss = 0
                        for s in spline:
                            if metric in s:
                                tloss = s.split(metric)[-1]
                                break
                        if tloss == 0:
                            raise ValueError('This metric is not found for validation')
                        # tloss = float(tloss.split(' ')[0])

                    else:
                        if bool_iter_epoch:
                            iter_epoch += 1
                        try:
                            tloss = 0
                            for s in spline:
                                if metric in s:
                                    tloss = s.split(metric)[-1]

                                    break
                            if tloss == 0:
                                raise ValueError('This metric is not found for validation')
                            tloss = float(tloss.split('\b')[0])
                        except:
                            tloss = float('nan')

                    train_l.append(tloss)


    train_t = list(range(len(train_l)))
    val_t = list(range(int(iter_epoch), len(train_l) + 1,int(iter_epoch)))

    return train_t, val_t, train_l, val_l


def create_results_dir(dir_path):
    if not exists(dir_path):
        os.makedirs(dir_path)
    if not exists(join(dir_path,'model_docs')):
        os.makedirs(join(dir_path,'model_docs'))
    if not exists(join(dir_path,'model_weights')):
        os.makedirs(join(dir_path,'model_weights'))
    if not exists(join(dir_path,'logs')):
        os.makedirs(join(dir_path,'logs'))
    if not exists(join(dir_path, 'results')):
        os.makedirs(join(dir_path, 'results'))


