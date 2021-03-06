import nibabel as nib
import numpy as np
import os
from os.path import join
from src.utils.preprocessing import flip_plane, normalize_image
from scipy.ndimage.morphology import binary_closing


to_int = lambda b: 1 if b else 0

class Subject:
    def __init__(self,
                 data_path,
                 id):

        self.data_path = data_path
        self._id = id

        self.subject_filepath = join(self.data_path, self.id)

        self.FLAIR_FILE = join(self.subject_filepath,'pre','FLAIR.nii.gz')
        self.T1_FILE = join(self.subject_filepath,'pre','T1.nii.gz')
        self.GT_FILE = join(self.subject_filepath,'wmh.nii.gz')
        self.ROIMASK_FILE = join(self.subject_filepath,'pre','mask_closing.nii.gz')

        self.data_augmentation = False

    @property
    def id(self):
        return self._id

    def get_affine(self):
        return nib.load(self.GT_FILE).affine

    def load_channels(self,normalize = False):
        modalities = []
        modalities.append(nib.load(self.FLAIR_FILE))
        modalities.append(nib.load(self.T1_FILE))

        channels = np.zeros( modalities[0].shape + (2,), dtype=np.float32)

        for index_mod, mod in enumerate(modalities):
            if self.data_augmentation:
                channels[:, :, :, index_mod] = flip_plane(np.asarray(mod.dataobj))
            else:
                channels[:,:,:,index_mod] = np.asarray(mod.dataobj)

            if normalize:
                channels[:, :, :, index_mod] = normalize_image(channels[:,:,:,index_mod], mask = self.load_ROI_mask() )


        return channels

    def load_labels(self):
        labels_proxy = nib.load(self.GT_FILE)
        if self.data_augmentation:
            labels = flip_plane(np.asarray(labels_proxy.dataobj))
        else:
            labels = np.asarray(labels_proxy.dataobj)
        labels[np.where(labels == 2)] = 0 # To get rid of other pathologies
        return labels

    def load_ROI_mask(self):

        proxy = nib.load(self.FLAIR_FILE)
        image_array = np.asarray(proxy.dataobj)

        mask = np.ones_like(image_array)
        mask[np.where(image_array < 90)] = 0

        # img = nib.Nifti1Image(mask, proxy.affine)
        # nib.save(img, join(modalities_path,'mask.nii.gz'))

        struct_element_size = (20, 20, 20)
        mask_augmented = np.pad(mask, [(21, 21), (21, 21), (21, 21)], 'constant', constant_values=(0, 0))
        mask_augmented = binary_closing(mask_augmented, structure=np.ones(struct_element_size, dtype=bool)).astype(
            np.int)

        return mask_augmented[21:-21, 21:-21, 21:-21].astype('bool')

    def get_subject_shape(self):

        proxy = nib.load(self.T1_FILE)
        return proxy.shape

