from __future__ import print_function
from os.path import join
import h5py
import numpy as np
import argparse
import sys
from matplotlib import pyplot as plt
plt.switch_backend('TKAgg')


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.
    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path,'r')
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

            # print("    "+str(layer))
            for p_name in g.keys():
                # if 'batch' in p_name or 'input' in p_name:
                if 'input' in p_name:
                    continue

                param = g[p_name]
                if len(param) > 0:
                    param = param[p_name]
                    if 'batch_normalization' in param.name:
                        continue

                    for k,v in param.items():
                        print("      {}: {}".format(k, v.shape))
                        print(np.mean(np.asarray(v.value).flatten()))
                        print(np.std(np.asarray(v.value).flatten()))
                        print(v.value)
                        plt.hist(np.asarray(v.value).flatten())
                        plt.show()
                        plt.close()
    except:
        print("Error")
    finally:
        f.close()


if __name__ == "__main__":
    # Parse arguments from command-line

    file = ''
    w_file_path = join('', file)

    print_structure(w_file_path)