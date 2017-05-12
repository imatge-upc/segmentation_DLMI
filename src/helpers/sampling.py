
import numpy as np
from src.helpers.preprocessing_utils import padding3D


def sampling_scheme_selection(sampling_scheme,n_classes):
    if sampling_scheme == 'uniform':
        return UniformScheme(n_classes)
    elif sampling_scheme == 'foreground-background':
        return FBScheme(n_classes)
    elif sampling_scheme == 'whole':
        return WholeScheme(n_classes)
    elif sampling_scheme == "per-class":
        return PerClassScheme(n_classes)
    else:
        raise ValueError("SamplingScheme error: Please, specify a correct sampling scheme")


#ROI mask means region of interest in overall. Otherwise, the whole image is taken
#GT labels is used to take the foreground or class masks

class SamplingScheme(object):

    def __init__(self,n_classes):
        self.n_classes = n_classes


    def set_probabilities(self, weights):
        raise NotImplementedError

    def get_weighted_mask(self,ROI_mask, labels_mask, ):
        raise NotImplementedError

    def get_probabilities(self):
        raise NotImplementedError

    def normalize(self, weights):
        weights = np.asarray(weights, dtype = "int32")
        return weights / (1.0*np.sum(weights))

    def get_mask_boundaries(self,image_shape,mask_shape,ROI_mask):
        half_segment_dimensions = np.zeros((len(image_shape), 2), dtype='int32')
        for index, dim in enumerate(image_shape):
            if dim % 2 == 0:
                half_segment_dimensions[index, :] = [dim / 2 - 1, dim / 2]
            else:
                half_segment_dimensions[index, :] = [np.floor(dim / 2)] * 2

        mask_boundaries = np.zeros(mask_shape, dtype='int32')
        mask_boundaries[half_segment_dimensions[0][0]:-half_segment_dimensions[0][1],
        half_segment_dimensions[1][0]:-half_segment_dimensions[1][1],
        half_segment_dimensions[2][0]:-half_segment_dimensions[2][1]] = 1

        if ROI_mask is None:
            return  mask_boundaries
        else:
            return mask_boundaries * ROI_mask

    def get_proper_images(self,input_shape,input,labels,roi):
        if roi is not None:
            labels = labels * roi
            input = input * np.reshape(roi, (1,) + roi.shape)
        return input,labels,roi

class UniformScheme(SamplingScheme):

    def __init__(self,n_classes):
        self.n_categories = 1
        super(UniformScheme, self).__init__(n_classes)


    def set_probabilities(self,weights):
        self._probability = [1.0]

    def get_weighted_mask(self,image_shape, mask_shape,ROI_mask=None, labels_mask=None):

        mask_boundaries = self.get_mask_boundaries(image_shape,mask_shape,ROI_mask)[np.newaxis,:]

        if ROI_mask is not None:
            return ROI_mask[np.newaxis,:] * mask_boundaries
        else:
            return mask_boundaries

    def get_probabilities(self):
        return self._probability



class WholeScheme(SamplingScheme):

    def __init__(self,n_classes):
        self.n_categories = 1
        super(WholeScheme, self).__init__(n_classes)

    def set_probabilities(self,weights):
        self._probability = [1.0]

    def get_weighted_mask(self,image_shape, mask_shape,ROI_mask=None, labels_mask=None):

        mask = np.zeros((1,)+mask_shape)
        central_voxel = [0]
        for index, dim in enumerate(image_shape):
            if dim % 2 == 0:
                central_voxel.append( dim / 2 -1)
            else:
                central_voxel.append(np.floor(dim / 2))
        mask[tuple(central_voxel)] = 1
        print 'sampling'
        print np.where(mask == 1)
        return mask

    def get_probabilities(self):
        return self._probability

    def get_proper_images(self,input_shape,input,labels,roi):

        input = padding3D(input,'match', input_shape)
        labels = padding3D(labels, 'match', input_shape)
        if roi is not None:
            roi = padding3D(roi, 'match', input_shape)

        return super(WholeScheme,self).get_proper_images(input_shape,input,labels,roi)


class FBScheme(SamplingScheme):

    def __init__(self,n_classes):
        self.n_categories = 2
        super(FBScheme, self).__init__(n_classes)

    def set_probabilities(self,weights):
        assert len(weights) == self.n_categories
        self._probability = self.normalize(weights)

    def get_weighted_mask(self, image_shape, mask_shape,ROI_mask=None, labels_mask=None):

        if labels_mask == None:
            raise ValueError('SamplingScheme error: please specify a labels_mask for this sampling scheme')

        mask_boundaries = self.get_mask_boundaries(image_shape, mask_shape, ROI_mask)
        final_mask = np.zeros((2,) + labels_mask.shape, dtype="int16")
        final_mask[0,:] = (labels_mask > 0) * mask_boundaries
        final_mask[1,:] = (final_mask[0]==0) * mask_boundaries
        final_mask = 1.0 * final_mask / np.reshape(np.sum(np.reshape(final_mask,(2, -1)), axis=1),(2,)+(1,)*len(image_shape))

        return final_mask


    def get_probabilities(self):
        return self._probability



class PerClassScheme(SamplingScheme):

    def __init__(self,n_classes):
        self.n_categories = n_classes
        super(PerClassScheme, self).__init__(n_classes)


    def set_probabilities(self, weights):
        assert len(weights) == self.n_categories
        self._probability = self.normalize(weights)


    def get_weighted_mask(self, image_shape, mask_shape,ROI_mask=None, labels_mask=None):

        if labels_mask == None:
            raise ValueError('SamplingScheme error: please specify a labels_mask for this sampling scheme')

        mask_boundaries = self.get_mask_boundaries(image_shape, mask_shape,ROI_mask)


        final_mask = np.zeros((self.n_categories,) + labels_mask.shape, dtype="int16")
        for index_cat in self.n_categories:
            final_mask[index_cat] = (labels_mask == index_cat,) * mask_boundaries

        final_mask = 1.0 * final_mask / np.reshape(np.sum(np.reshape(final_mask,(self.n_categories,-1)),axis=1),(self.n_categories,)+(1,)*len(image_shape))

        return final_mask

    def get_probabilities(self):
        return self._probability
