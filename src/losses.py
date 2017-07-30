import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error


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
    # cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
    cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy

def categorical_crossentropy_3d_masked(vectors):
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

    y_predicted, mask, y_true = vectors

    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_predicted)
    y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
    num_total_elements = K.sum(mask)
    # cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
    cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy



def dice_cost(y_true, y_predicted):

    mask_true = K.flatten(y_true[:, :, :, :, 1])#
    mask_pred = K.flatten(y_predicted[:, :, :, :, 1])#

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return -num_sum/den_sum

def dice_cost_123(y_true, y_predicted):

    dice_1 = dice_cost(y_true, y_predicted)
    dice_2 = dice_cost_2(y_true, y_predicted)
    dice_3 = dice_cost_3(y_true, y_predicted)


    return 1/3*(dice_1+dice_2+dice_3)

def dice_cost_2(y_true, y_predicted):

    mask_true = K.flatten(y_true[:, :, :, :, 2])#
    mask_pred = K.flatten(y_predicted[:, :, :, :, 2])#

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return -num_sum/den_sum

def dice_cost_3(y_true, y_predicted):

    mask_true = K.flatten(y_true[:, :, :, :, 3])#
    mask_pred = K.flatten(y_predicted[:, :, :, :, 3])#

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return -num_sum/den_sum


def scae_mean_squared_error_masked(vectors):

    y_true,y_pred,mask = vectors
    return 1/K.sum(mask, axis=[0, 1, 2, 3, 4])*K.sum(K.square(y_pred - y_true), axis=[0, 1, 2, 3, 4])


def mean_squared_error_lambda(vectors):
    y_true, y_pred = vectors
    return mean_squared_error(y_true, y_pred)

def categorical_crossentropy_3d_lambda(vectors):
    y_true, y_pred = vectors

    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())

    # cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
    cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (K.sum(y_true) + K.epsilon())
    return mean_cross_entropy