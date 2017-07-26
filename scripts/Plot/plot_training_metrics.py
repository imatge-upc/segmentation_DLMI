import argparse
import sys

import matplotlib.pyplot as plt

import params as p
import scripts.Plot as m
from src.utils.io import get_loss_lines

if __name__ == "__main__":
    """ CONSTANTS """

    # Switch backend
    plt.switch_backend('Agg')

    # Parse arguments from command-line
    arguments_parser = argparse.ArgumentParser()
    arguments_parser.add_argument('-metric', help='Metric to be plotted')
    arguments_parser.add_argument('-log', help='Path to the log file to be loaded')
    args = arguments_parser.parse_args(sys.argv[1:])

    metric = args.metric
    log_file = args.log

    t_train, t_val, train_losses, val_losses = get_loss_lines(log_file, metric=metric)

    print('Number of train metrics: {}'.format(len(train_losses)))
    print('Number of validation metrics: {}'.format(len(val_losses)))
    print('Number of validation values: {}'.format(len(t_val)))

    train = plt.plot(t_train, train_losses, 'b-', label='Training ' + metric)
    val = plt.plot(t_val, val_losses, 'r-*', label='Validation ' + metric)

    plt.xlabel('iteration')
    plt.ylabel(metric)

    # plt.axis([0, t_train[-1], 0, max(train_losses)*(1+0.1)])
    plt.axis([0, t_train[-1], 0,1])
    plt.legend(loc=3, bbox_to_anchor=(0, 1.05),
              ncol=3, fancybox=True, shadow=True, prop={'size': 6})
    plt.savefig(log_file[:-3]+'_'+metric+'.png')