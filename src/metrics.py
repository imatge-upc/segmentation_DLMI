import keras.backend as K
import tensorflow as tf
import numpy as np


def dice(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric
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
    y_pred_decision = tf.floor((y_pred + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

    y_sum = K.sum(y_true * y_pred_decision)

    return (2. * y_sum + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred_decision) + K.epsilon())


def accuracy(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=4), K.argmax(y_pred, axis=4)))


def dice_coef(y_true, y_pred):

    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred_decision)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())



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

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))
    print(np.shape(y_pred_decision))

    mask_true = K.sum(y_true[:, :, :, :,1:3], axis=4)
    mask_pred = K.sum(y_pred_decision[:, :, :, :, 1:3], axis=4)

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

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

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

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 3]
    mask_pred = y_pred_decision[:, :, :, :, 3]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())

# def accuracy_survival(y_true, y_predicted):



def dice_whole_mod(y_true, y_pred):
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
    # mask = K.expand_dims(K.sum(y_true,axis=4),axis=4)
    # cmp_mask = K.concatenate([K.ones_like(mask) - mask,K.zeros_like(mask), K.zeros_like(mask)],axis=4)
    # y_pred = y_pred + cmp_mask

    y_true = y_true[:,:,:,:,:3]
    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

    mask_true = K.sum(y_true, axis=4)
    mask_pred = K.sum(y_pred_decision, axis=4) * K.sum(y_true, axis=4)

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_core_mod(y_true, y_pred):
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
    y_true = y_true[:,:,:,:,:3]


    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))
    y_pred_decision = tf.where(tf.is_nan(y_pred_decision), tf.zeros_like(y_pred_decision), y_pred_decision)


    mask_true1 = K.expand_dims(y_true[:, :, :, :, 2],axis=4)
    mask_true2 = K.expand_dims(y_true[:, :, :, :, 0],axis=4)
    mask_true = K.sum(K.concatenate([mask_true1, mask_true2], axis=4), axis=4)
    mask_pred1 = K.expand_dims(y_pred_decision[:, :, :, :, 2],axis=4)
    mask_pred2 = K.expand_dims(y_pred_decision[:, :, :, :, 0],axis=4)
    mask_pred = K.sum(K.concatenate([mask_pred1, mask_pred2], axis=4), axis=4) * K.sum(y_true, axis=4)

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_enhance_mod(y_true, y_pred):
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
    y_true = y_true[:,:,:,:,:3]

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))
    # y_pred_decision = tf.where(tf.is_nan(y_pred_decision), tf.zeros_like(y_pred_decision), y_pred_decision)



    mask_true = y_true[:, :, :, :, 2]
    mask_pred = y_pred_decision[:, :, :, :, 2] * K.sum(y_true, axis=4)

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())





def dice_0(y_true, y_pred):
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

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :,0]
    mask_pred = y_pred_decision[:, :, :, :, 0]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_1(y_true, y_pred):
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

    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :,1]
    mask_pred = y_pred_decision[:, :, :, :, 1]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())

def dice_1_2D(y_true, y_pred):
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

    y_pred_decision = tf.floor((y_pred  + K.epsilon())/ K.max(y_pred, axis=2, keepdims=True))

    mask_true = y_true[:, :, 1]
    mask_pred = y_pred_decision[:, :, 1]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_2(y_true, y_pred):
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

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :, 2]
    mask_pred = y_pred_decision[:, :, :, :, 2]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_3(y_true, y_pred):
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
    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :, 3]
    mask_pred = y_pred_decision[:, :, :, :, 3]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_4(y_true, y_pred):
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

    y_pred_decision = tf.floor((y_pred  + K.epsilon()) / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :, 4]
    mask_pred = y_pred_decision[:, :, :, :, 4]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())





def precision_0(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 0]
    mask_pred = y_pred_decision[:, :, :, :, 0]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())


def precision_1(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 1]
    mask_pred = y_pred_decision[:, :, :, :, 1]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())


def precision_2(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 2]
    mask_pred = y_pred_decision[:, :, :, :, 2]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())


def precision_3(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 3]
    mask_pred = y_pred_decision[:, :, :, :, 3]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())


def precision_4(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 4]
    mask_pred = y_pred_decision[:, :, :, :, 4]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())


def recall_0(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 0]
    mask_pred = y_pred_decision[:, :, :, :, 0]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_true) + K.epsilon())


def recall_1(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 1]
    mask_pred = y_pred_decision[:, :, :, :, 1]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_true) + K.epsilon())


def recall_2(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 2]
    mask_pred = y_pred_decision[:, :, :, :, 2]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_true) + K.epsilon())


def recall_3(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 3]
    mask_pred = y_pred_decision[:, :, :, :, 3]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_true) + K.epsilon())


def recall_4(y_true, y_pred):
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))
    mask_true = y_true[:, :, :, :, 4]
    mask_pred = y_pred_decision[:, :, :, :, 4]

    y_sum = K.sum(mask_true * mask_pred)

    return (y_sum + K.epsilon()) / (K.sum(mask_true) + K.epsilon())


# -------------------------- Masked metrics --------------------------------

def accuracy_mask(mask):
    def accuracy(y_true, y_pred):
        intersection = K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1))
        intersection_masked = mask * intersection
        return 1.0 * K.sum(intersection_masked) / K.sum(mask)

    return accuracy


def dice_whole_mask(mask):
    def dice_whole_closure(y_true, y_pred):
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

        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = K.sum(y_true[:, [1, 2, 3, 4], :, :, :], axis=1)
        mask_pred = K.sum(y_pred_decision[:, [1, 2, 3, 4], :, :, :], axis=1)

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (2. * y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.sum(mask * mask_pred) + K.epsilon())

    return dice_whole_closure


def dice_core_mask(mask):
    def dice_core_closure(y_true, y_pred):
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

        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = K.sum(y_true[:, [1, 3, 4], :, :, :], axis=1)
        mask_pred = K.sum(y_pred_decision[:, [1, 3, 4], :, :, :], axis=1)

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (2. * y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.sum(mask * mask_pred) + K.epsilon())

    return dice_core_closure


def dice_enhance_mask(mask):
    def dice_enhance_closure(y_true, y_pred):
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

        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 4, :, :, :]
        mask_pred = y_pred_decision[:, 4, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (2. * y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.sum(mask * mask_pred) + K.epsilon())

    return dice_enhance_closure


def precision_0_mask(mask):
    def precision_0_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 0, :, :, :]
        mask_pred = y_pred_decision[:, 0, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())

    return precision_0_closure


def precision_1_mask(mask):
    def precision_1_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 1, :, :, :]
        mask_pred = y_pred_decision[:, 1, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())

    return precision_1_closure


def precision_2_mask(mask):
    def precision_2_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 2, :, :, :]
        mask_pred = y_pred_decision[:, 2, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())

    return precision_2_closure


def precision_3_mask(mask):
    def precision_3_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 3, :, :, :]
        mask_pred = y_pred_decision[:, 3, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())

    return precision_3_closure


def precision_4_mask(mask):
    def precision_4_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 4, :, :, :]
        mask_pred = y_pred_decision[:, 4, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask_pred) + K.epsilon())

    return precision_4_closure


def recall_0_mask(mask):
    def recall_0_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 0, :, :, :]
        mask_pred = y_pred_decision[:, 0, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.epsilon())

    return recall_0_closure


def recall_1_mask(mask):
    def recall_1_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 1, :, :, :]
        mask_pred = y_pred_decision[:, 1, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.epsilon())

    return recall_1_closure


def recall_2_mask(mask):
    def recall_2_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 2, :, :, :]
        mask_pred = y_pred_decision[:, 2, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.epsilon())

    return recall_2_closure


def recall_3_mask(mask):
    def recall_3_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 3, :, :, :]
        mask_pred = y_pred_decision[:, 3, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.epsilon())

    return recall_3_closure


def recall_4_mask(mask):
    def recall_4_closure(y_true, y_pred):
        y_pred_decision = K.cast(y_pred / K.max(y_pred, axis=1, keepdims=True), 'int8')
        mask_true = y_true[:, 4, :, :, :]
        mask_pred = y_pred_decision[:, 4, :, :, :]

        y_sum = K.sum(mask * mask_true * mask_pred)

        return (y_sum + K.epsilon()) / (K.sum(mask * mask_true) + K.epsilon())

    return recall_4_closure
