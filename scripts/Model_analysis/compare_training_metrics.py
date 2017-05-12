import argparse
import sys
import matplotlib.pyplot as plt

import params as p
from src.helpers.io_utils import get_loss_lines
import scripts.Model_analysis as m

if __name__ == "__main__":
    """ CONSTANTS """


    # Switch backend
    plt.switch_backend('Agg')

    # Parse arguments from command-line
    arguments_parser = argparse.ArgumentParser()
    arguments_parser.add_argument('-model', help='Name of the model to be used', choices=p.PARAMS_DICT.keys())
    arguments_parser.add_argument('-metric', help='Metric to be plotted', choices=m.METRICS_DICT.keys())
    arguments_parser.add_argument('-log1', help='Path to the log file to be loaded')
    arguments_parser.add_argument('-log2', help='Path to the log file to be loaded')
    args = arguments_parser.parse_args(sys.argv[1:])

    model_name = args.model
    metric = args.metric
    log_file_1 = args.log1
    log_file_2 = args.log2

    params = p.PARAMS_DICT[model_name].get_params()
    train_losses1, val_losses1 = get_loss_lines(log_file_1, metric=metric, metric_dictionary=m.METRICS_DICT_U_NET)
    train_losses2, val_losses2 = get_loss_lines(log_file_2, metric=metric, metric_dictionary=m.METRICS_DICT)

    t_train1 = range(len(train_losses1))
    t_train2 = range(len(train_losses2)*5/10)

    print len(train_losses2[::10/5])
    print len(t_train2)
    plt.figure()
    plt.plot(t_train1[:len(t_train2)], train_losses1[:len(t_train2)], 'b-', label='Multi-resolution model ' + metric)
    plt.plot(t_train2, train_losses2[::10/5][:len(t_train2)], 'r-.', label='Simple model ' + metric)
    plt.ylim(([0, 1.2]))
    plt.xlabel('iteration')
    plt.ylabel(metric)
    # plt.legend(loc=3, bbox_to_anchor=(0, 1.05),
    #            ncol=3, fancybox=True, shadow=True, prop={'size': 6})
    plt.savefig(model_name + '_' + metric + '_train.png')


    iter_epoch1 = params[p.N_SEGMENTS_TRAIN] / 20  # how many iterations does an epoch have
    iter_epoch2 = params[p.N_SEGMENTS_TRAIN] / params[p.BATCH_SIZE]
    t_val1 = range(int(iter_epoch1), len(train_losses1) + 1, int(iter_epoch1))[:len(val_losses1)]  # validation loss is displayed after each epoch only
    t_val2 = range(int(iter_epoch2), len(train_losses2) + 1, int(iter_epoch2))[:len(val_losses2)]
    t_val1 = range(len(val_losses1))
    t_val2 = range(len(val_losses2))

    plt.figure()
    plt.plot(t_val1[:len(val_losses2)], val_losses1[:len(val_losses2)], 'b-', label='Multi-resolution model ' + metric)
    plt.plot(t_val2, val_losses2, 'r-*', label='Simple model ' + metric)
    plt.ylim(([0, 1.2]))
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    # plt.legend(loc=3, bbox_to_anchor=(0, 1.05),
    #            ncol=3, fancybox=True, shadow=True, prop={'size': 6})
    plt.savefig(model_name + '_' + metric + '_val.png')