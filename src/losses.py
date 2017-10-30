import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error


M_tree_5 = np.array([[0., 1., 1., 1., 1.],
                   [1., 0., 0.6, 0.2, 0.5],
                   [1., 0.6, 0., 0.6, 0.7],
                   [1., 0.2, 0.6, 0., 0.5],
                   [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)

M_tree_4 = np.array([[0., 1., 1., 1.,],
                     [1., 0., 0.6, 0.5],
                     [1., 0.6, 0., 0.7],
                     [1., 0.5, 0.7, 0.]], dtype=np.float64)


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


def categorical_crossentropy_3d_SW(y_true_sw, y_predicted):
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
    sw = y_true_sw[:,:,:,:,K.int_shape(y_predicted)[-1]:]
    y_true = y_true_sw[:,:,:,:,:K.int_shape(y_predicted)[-1]]

    y_true_flatten = K.flatten(y_true*sw)
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

    num_sum = 2.0 * K.sum(y_true * y_predicted) + K.epsilon()
    den_sum = K.sum(y_true) + K.sum(y_predicted)+ K.epsilon()

    return -num_sum/den_sum

def dice_cost_1(y_true, y_predicted):

    mask_true = y_true[:, :, :, :, 1]
    mask_pred = y_predicted[:, :, :, :, 1]

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return -num_sum/den_sum

def dice_cost_123(y_true, y_predicted):

    dice_1 = dice_cost_1(y_true, y_predicted)
    dice_2 = dice_cost_2(y_true, y_predicted)
    dice_3 = dice_cost_3(y_true, y_predicted)


    return 1/3*(dice_1+dice_2+dice_3)

def dice_cost_12(y_true, y_predicted):

    dice_1 = dice_cost_1(y_true, y_predicted)
    dice_2 = dice_cost_2(y_true, y_predicted)


    return 1/2*(dice_1+dice_2)

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


def wasserstein_disagreement_map(prediction, ground_truth, M):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened pred_proba and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = K.int_shape(prediction)[-1]
    # unstack_labels = tf.unstack(ground_truth, axis=-1)
    ground_truth = tf.cast(ground_truth, dtype=tf.float64)
    # unstack_pred = tf.unstack(prediction, axis=-1)
    prediction = tf.cast(prediction, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(prediction[:,i], ground_truth[:,j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


def generalised_wasserstein_dice_loss(y_true, y_predicted ):


    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    # apply softmax to pred scores
    n_classes = K.int_shape(y_predicted)[-1]


    ground_truth = tf.cast(tf.reshape(y_true,(-1,n_classes)), dtype=tf.int64)
    pred_proba = tf.cast(tf.reshape(y_predicted,(-1,n_classes)), dtype=tf.float64)

    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree_4
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(pred_proba, ground_truth, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(ground_truth, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)

    return tf.cast(WGDL, dtype=tf.float32)

