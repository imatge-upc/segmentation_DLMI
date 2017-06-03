import keras.backend as K


def elementwise_softmax_3d(matrix):
    """
    Computes element-wise softmax for 3D arrays (volumes), that is, for a matrix with shape
    (num_samples, num_classes, dim1, dim2, dim3)

    Parameters
    ----------
    matrix : keras.placeholder
        Placeholder for the 3D array whose softmax distribution we want to compute

    Returns
    -------
    keras.placeholder
        Placeholder for a 3D array with the softmax distribution for all classes with shape
        (num_samples, dim1, dim2, dim3, num_classes)

    """
    expon = lambda x: K.exp(x)
    expon_matrix = expon(matrix)
    softmax_matrix = expon_matrix / K.sum(expon_matrix, axis=4, keepdims=True)
    return softmax_matrix
