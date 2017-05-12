import numpy as np


def one_hot_representation(y_data, number_classes):
    """
    Given a n-dimensional matrix representing the ground truth labels, with the following
    structure (n_samples, dim1, dim2, ..., dimN) and containing values between 0 and number_classes7
    (representing the encoded classes), return a one-hot representation of such matrix

    Parameters
    ----------
    y_data : numpy.array
        N-dimensional matrix with shape (n_samples, dim1, dim2, ..., dimN) representing each label with a
        value between 0 and number_classes - 1
    number_classes : int
        The number of classes labeled in the data

    Returns
    -------
    numpy.array
        N+1 dimensional array with shape (n_samples, number_classes, dim1, dim2, ..., dimN) that is the
        hot-encoding representation of the original y_data array.
    """
    if isinstance(y_data,list):
        y_data = np.asarray(y_data)
    shape = y_data.shape
    result = np.zeros((shape[0], number_classes) + shape[1:])
    for i in range(number_classes):
        index = np.where(y_data == i)
        one_hot_index = (index[0], i) + index[1:]
        result[one_hot_index] = 1
    return np.asarray(result, dtype=np.float64)


def padding3D(input, width_mode, pad_factor):

    if width_mode == 'multiple':
        assert isinstance(pad_factor, int)
        shape = input.shape[-3:]
        added_shape = [(0,0)]*len(input.shape[:-3])
        for dim in shape:
            added_shape.append((0,dim % pad_factor))
        output = np.pad(input, tuple(added_shape), 'constant', constant_values=(0, 0))

    elif width_mode == 'fixed':
        assert isinstance(pad_factor,list) or isinstance(pad_factor,tuple)
        output = np.pad(input, tuple(pad_factor), 'constant',constant_values=(0, 0))

    elif width_mode == 'match':
        assert isinstance(pad_factor, list) or isinstance(pad_factor, tuple)
        shape = input.shape[-3:]
        shape_difference = np.asarray(pad_factor) - np.asarray(shape)
        added_shape = [(0, 0)] * len(input.shape[:-3])
        subs_shape = [np.s_[:]]* len(input.shape[:-3])
        for diff in shape_difference:
            if diff < 0:
                subs_shape.append(np.s_[:diff])
                added_shape.append((0, 0))
            else:
                subs_shape.append(np.s_[:])
                added_shape.append((0, diff))

        output = np.pad(input, tuple(added_shape), 'constant', constant_values=(0, 0))
        output = output[subs_shape]
    else:
        raise ValueError("Padding3D error (src.helpers.preprocessing_utils): No existen padding method " + str(width_mode))
    return output


def _bias_correction(self, image_patch):
    # if not isinstance(image_patch, list):
    #     [image_patch]
    #
    # for patch in image_patch:
    #     corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #     output = corrector.Execute(patch, maskImage)
    raise NotImplementedError


def _normalization(self):
    raise NotImplementedError
