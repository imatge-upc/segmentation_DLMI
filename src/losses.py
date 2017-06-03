import keras.backend as K
import numpy as np
import tensorflow as tf
from src.metrics import dice_whole


def categorical_crossentropy_3d(y_true, y_predicted):
    """
    Computes categorical cross-entropy loss for a softmax distribution in a hot-encoded 3D array
    with shape (num_samples, num_classes, dim1, dim2, dim3)

    Parameters
    ----------
    y_true : keras.placeholder [batches, dim0,dim1,dim2]
        Placeholder for data holding the ground-truth labels encoded in a one-hot representation
    y_predicted : keras.placeholder [batches,channels,dim0,dim1,dim2]
        Placeholder for data holding the softmax distribution over classes

    Returns
    -------
    scalar
        Categorical cross-entropy loss value
    """
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_predicted)
    y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
    num_total_elements = K.sum(y_true_flatten)
    #cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
    cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy


def categorical_crossentropy_3d_mask(mask):
    """
    Computes categorical cross-entropy loss for a softmax distribution in a hot-encoded 3D array
    with shape (num_samples, num_classes, dim1, dim2, dim3)

    Parameters
    ----------
    y_true : keras.placeholder [batches, dim0,dim1,dim2]
        Placeholder for data holding the ground-truth labels encoded in a one-hot representation
    y_predicted : keras.placeholder [batches,channels,dim0,dim1,dim2]
        Placeholder for data holding the softmax distribution over classes

    Returns
    -------
    scalar
        Categorical cross-entropy loss value
    """

    def masked_cross_entropy_3D(y_true, y_predicted):
        mask_redim = K.expand_dims(mask, 1)
        y_true_flatten = K.flatten(y_true * mask_redim)
        y_pred_flatten = K.flatten(y_predicted)

        y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
        num_total_elements = K.sum(mask)
        cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
        mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())

        return mean_cross_entropy

    return masked_cross_entropy_3D


def dice_cost(y_true, y_predicted):
    return -dice_whole(y_true, y_predicted)
