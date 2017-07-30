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
    result = np.zeros((shape[0],) + shape[1:]+(number_classes,))
    for i in range(number_classes):
        index = np.where(y_data == i)
        one_hot_index = (index[0], ) + index[1:] + (i,)
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

def resize_image(image,target_shape, pad_value = 0):
    assert isinstance(target_shape, list) or isinstance(target_shape, tuple)
    add_shape, subs_shape = [], []

    image_shape = image.shape
    shape_difference = np.asarray(target_shape, dtype=int) - np.asarray(image_shape,dtype=int)
    for diff in shape_difference:
        if diff < 0:
            subs_shape.append(np.s_[int(np.abs(np.ceil(diff/2))):int(np.floor(diff/2))])
            add_shape.append((0, 0))
        else:
            subs_shape.append(np.s_[:])
            add_shape.append((int(np.ceil(1.0*diff/2)),int(np.floor(1.0*diff/2))))
    output = np.pad(image, tuple(add_shape), 'constant', constant_values=(pad_value, pad_value))
    output = output[subs_shape]
    return output

def scale_image(image, max_value=None):

    if max_value == None:
        max_value = np.max(image)
    if max_value == 0:
        warnings.warn("The image is constant. Please, check it out!")

    output_image = 1.0*image/max_value

    return output_image

def compute_class_frequencies(segment,num_classes):
    if isinstance(segment,list):
        segment = np.asarray(segment)
    f = 1.0 * np.bincount(segment.reshape(-1,).astype(int),minlength=num_classes) / np.prod(segment.shape)
    return f

def compute_centralvoxel_frequencies(segment,minlength):
    if isinstance(segment,list):
        segment = np.asarray(segment)
    shape = segment.shape[-3:]

    middle_coordinate = np.zeros(3,int)
    for it_coordinate,coordinate in enumerate(shape):
        if coordinate%2==0:
            middle_coordinate[it_coordinate] = coordinate / 2 - 1
        else:
            middle_coordinate[it_coordinate] = coordinate/2

    segment = segment.reshape((-1,) + shape)
    f = 1.0 * np.bincount(segment[:,middle_coordinate[0],middle_coordinate[1],middle_coordinate[2]].reshape(-1,).astype(int),minlength=minlength) / np.prod(segment.shape[:-3])
    return f


def _bias_correction(image_patch):
    # if not isinstance(image_patch, list):
    #     [image_patch]
    #
    # for patch in image_patch:
    #     corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #     output = corrector.Execute(patch, maskImage)
    raise NotImplementedError


def normalize_image(image, mask=None):

    if mask is not None:
        image = image*mask
        mean = np.mean(image)#np.sum(image) / np.sum(mask)
        std = np.sqrt( 1 / (np.sum(mask)-1) * np.sum((image - mean) ** 2))
    else:
        mean = np.mean(image)
        std = np.std(image)

    return (image-mean)/std

def flip_plane(array,plane=0):
    # Flip axial plane LR, i.e. change left/right hemispheres. 3D tensors-only, batch_size=1.
    # n_slices = array.shape[2]
    # for i in range(n_slices):
    #     array[:,:,i] = np.flipud(array[:,:,i])
    # return array
    n_x = array.shape[plane]
    for i in range(n_x):
        if plane == 0:
            array[i,:,:] = np.flipud(array[i,:,:])
        if plane == 1:
            array[:,i,:] = np.flipud(array[:,i,:])
        if plane == 2:
            array[:,:,i] = np.flipud(array[:,:,i])
    return array
