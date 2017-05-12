from scipy.ndimage import binary_opening
import numpy as np

""" CONSTANTS """
BACKGROUND_CLASS = 2
MORPHOLOGICAL_STRUCTURE_SIZE = 8


def bypass(x):
    return x


def morphological_opening_background(x):
    # Compute an opening, which is the combination of an erosion and a dilation (morphological operators).
    for i in range(x.shape[0]):
        background_predictions = x[i, BACKGROUND_CLASS, :, :, :]
        binarized_predictions = np.array(background_predictions > 0, dtype=np.int32)
        opened_predictions = binary_opening(binarized_predictions, np.ones((MORPHOLOGICAL_STRUCTURE_SIZE,) * 3))
        x[i, BACKGROUND_CLASS, :, :, :] *= opened_predictions
    return x
