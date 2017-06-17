import argparse
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import params as p
from src.helpers.io_utils import get_loss_lines
import scripts.Model_analysis as m

if __name__ == "__main__":
    """ CONSTANTS """

    # Switch backend
    # plt.switch_backend('Qt4Agg')
    # print plt.get_backend()
    # Parse arguments from command-line
    arguments_parser = argparse.ArgumentParser()
    arguments_parser.add_argument('-params', help='Name of the model to be used', choices=p.PARAMS_DICT.keys())
    arguments_parser.add_argument('-metric', help='Metric to be plotted', choices=m.METRICS_DICT.keys())
    arguments_parser.add_argument('-log', help='Path to the log file to be loaded')
    args = arguments_parser.parse_args(sys.argv[1:])

    params_string = args.params
    metric = args.metric
    log_file = args.log

    params = p.PARAMS_DICT[params_string].get_params()
    train_losses, val_losses = get_loss_lines(log_file, metric=metric, metric_dictionary=m.METRICS_DICT)


    print 'Number of train metrics: {}'.format(len(train_losses))
    print 'Number of validation metrics: {}'.format(len(val_losses))

    t_train = range(len(train_losses))

    iter_epoch = params[p.N_SEGMENTS_TRAIN] / params[p.BATCH_SIZE]  # how many iterations does an epoch have
    t_val = range(int(iter_epoch), len(train_losses) + 1 ,
                  int(iter_epoch))[:len(val_losses)] # validation loss is displayed after each epoch only

    print 'Number of iterations per epoch (training): {}'.format(iter_epoch)
    print 'Number of validation values: {}'.format(len(t_val))

    print len(val_losses)
    print len(t_val)

    #print t_train
    #print train_losses
    #plt.plot(t_train)
    train = plt.plot(t_train, train_losses, 'b-*', label='Training ' + metric)
    val = plt.plot(t_val, val_losses, 'r-*', label='Validation ' + metric)


    plt.xlabel('iteration')
    plt.ylabel(metric)

    plt.legend(loc=3, bbox_to_anchor=(0, 1.05),
              ncol=3, fancybox=True, shadow=True, prop={'size': 6})
    plt.savefig(log_file[:-3]+'_'+metric+'.png')
