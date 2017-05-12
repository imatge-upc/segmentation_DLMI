from __future__ import print_function
from os.path import join
import h5py
import numpy as np
import argparse
import sys
from matplotlib import pyplot as plt


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(join('/work/acasamitjana/BRATS/output/model_weights',weight_file_path),'r')
    print(f.attrs.values())
    try:
        print('Starting ...')
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items()) == 0:
            print("Zero items")
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            print(g.keys())
            for p_name in g.keys():
                param = g[p_name]
                if 'batch' in p_name:
                    continue
                print("      {}: {}".format(p_name, param.shape))
                print(np.mean(np.asarray(param.value).flatten()))
                print(np.std(np.asarray(param.value).flatten()))
                # plt.hist(np.asarray(param.value).flatten())
                # plt.show()
    except:
        print("Error")
    finally:
        f.close()


if __name__ == "__main__":
    # Parse arguments from command-line
    arguments_parser = argparse.ArgumentParser()
    arguments_parser.add_argument('-w', help='Name of weights file')
    args = arguments_parser.parse_args(sys.argv[1:])

    w_file_path = args.w

    print_structure(w_file_path)
