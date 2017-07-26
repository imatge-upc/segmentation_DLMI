import keras.backend as K
import numpy as np



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

    y_predicted = K.concatenate([
    K.sum(K.concatenate([y_predicted[:,:,:,:,1:2],y_predicted[:,:,:,:,3:4]]),axis=4,keepdims=True),
        y_predicted[:,:,:,:,2:3],
        y_predicted[:,:,:,:,4:]
    ])
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_predicted)
    y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
    num_total_elements = K.sum(y_true_flatten)
    cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy

def dice_whole(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric, where P come from class 1,2,3,4,0
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : keras.placeholder
        Placeholder that contains the ground truth labels of the classes
    y_pred : keras.placeholder
        Placeholder that contains the class prediction

    Returns
    -------
    scalar
        Dice metric
    """
    print('shapes!')
    print(np.shape(y_pred))
    print(np.shape(y_true))

    y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=4, keepdims=True),'int8')
    print(np.shape(y_pred_decision))

    mask_true = K.sum(y_true[:, :, :, :,1:3], axis=4)
    mask_pred = K.sum(y_pred_decision[:, :, :, :, 1:], axis=4)

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_core(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric, where P come from class 1,2,3,4,5
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : keras.placeholder
        Placeholder that contains the ground truth labels of the classes
    y_pred : keras.placeholder
        Placeholder that contains the class prediction

    Returns
    -------
    scalar
        Dice metric
    """

    y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=4, keepdims=True),'int8')

    mask_true1 = y_true[:, :, :, :, 3:]
    mask_true2 = y_true[:, :, :, :, 1:2]
    mask_true = K.sum(K.concatenate([mask_true1, mask_true2], axis=4), axis=4)
    mask_pred1 = y_pred_decision[:, :, :, :, 3:]
    mask_pred2 = y_pred_decision[:, :, :, :, 1:2]
    mask_pred = K.sum(K.concatenate([mask_pred1, mask_pred2], axis=4), axis=4)

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_enhance(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric, where P come from class 1,2,3,4,5
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : keras.placeholder
        Placeholder that contains the ground truth labels of the classes
    y_pred : keras.placeholder
        Placeholder that contains the class prediction

    Returns
    -------
    scalar
        Dice metric
    """

    y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=4, keepdims=True),'int8')
    mask_true = y_true[:, :, :, :, 3]
    mask_pred = y_pred_decision[:, :, :, :, 3]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())
