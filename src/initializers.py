import keras.backend as K
import numpy as np


def bilinear_interpolation(shape, name=None):
    w = np.asarray([[[0.125, 0.25, 0.125], [0.25, 0.5, 0.25], [0.125, 0.25, 0.125]],
                    [[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]],
                    [[0.125, 0.25, 0.125], [0.25, 0.5, 0.25], [0.125, 0.25, 0.125]]])  # np.zeros(shape)
    return K.variable(w, name=name)
