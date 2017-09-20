from __future__ import print_function
from os.path import join
import csv
import numpy as np

from matplotlib import pyplot as plt
plt.switch_backend('TKAgg')


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.
    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    dice_WT = []
    dice_TC = []
    dice_ET = []


    with open(weight_file_path, 'rU') as csvfile:
        lines = csv.reader(csvfile, delimiter='\n')
        for line in lines:
            spline = line[0].split(':')
            if 'Dice WT' in spline:
                value = float(line[0].split('Dice WT: ')[-1])
                dice_WT.append(value)
            elif 'Dice TC' in spline:
                value = float(line[0].split('Dice TC: ')[-1])
                dice_TC.append(value)
            elif 'Dice ET' in spline:
                value = float(line[0].split('Dice ET: ')[-1])
                dice_ET.append(value)


        assert len(dice_WT) == len(dice_ET)
        assert len(dice_WT) == len(dice_TC)

        print('Dice WT: ' + str(np.sum(np.asarray(dice_WT))/len(dice_WT)))
        print('Dice TC: ' + str(np.sum(np.asarray(dice_TC))/len(dice_TC)))
        print('Dice ET: ' + str(np.sum(np.asarray(dice_ET)) / len(dice_ET)))

if __name__ == "__main__":

    w_file_path = '/imatge/acasamitjana/work/segmentation/BraTS/combi/20170815/LR_0.0005_mask_mask_seg/results/brats2017.txt'

    print_structure(w_file_path)