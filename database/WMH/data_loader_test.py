import nibabel as nib
import numpy as np
import os
from os.path import join
from src.utils.preprocessing import flip_plane, normalize_image
from scipy.ndimage.morphology import binary_closing


to_int = lambda b: 1 if b else 0

class Subject:
    def __init__(self,
                 data_path
                 ):

        self.data_path = data_path

        self.FLAIR_FILE = join(self.data_path,'pre','FLAIR.nii.gz')
        self.T1_FILE = join(self.data_path,'pre','T1.nii.gz')

        self.data_augmentation = False


    def get_affine(self):
        return nib.load(self.T1_FILE).affine

    def load_channels(self,normalize = False):
        modalities = []
        modalities.append(nib.load(self.FLAIR_FILE))
        modalities.append(nib.load(self.T1_FILE))

        channels = np.zeros( modalities[0].shape + (2,), dtype=np.float32)
        mask = self.load_ROI_mask()

        for index_mod, mod in enumerate(modalities):
            if self.data_augmentation:
                channels[:, :, :, index_mod] = flip_plane(np.asarray(mod.dataobj))
            else:
                channels[:,:,:,index_mod] = np.asarray(mod.dataobj)

            if normalize:
                channels[:, :, :, index_mod] = normalize_image(channels[:,:,:,index_mod], mask = mask )


        return channels

    def load_ROI_mask(self):


        proxy = nib.load(self.FLAIR_FILE)
        image_array = np.asarray(proxy.dataobj)
        image_array = normalize_image(image_array)


        mask = np.ones_like(image_array)
        mask[np.where(image_array < -0.3)] = 0

        struct_element_size = (20, 20, 20)
        mask_augmented = np.pad(mask, [(21, 21), (21, 21), (21, 21)], 'constant', constant_values=(0, 0))
        mask_augmented = binary_closing(mask_augmented, structure=np.ones(struct_element_size, dtype=bool)).astype(
            np.int)
        mask = mask_augmented[21:-21,21:-21,21:-21]

        if self.data_augmentation:
            mask = flip_plane(mask)

        return mask.astype('bool')


    def get_subject_shape(self):

        proxy = nib.load(self.T1_FILE)
        return proxy.shape



class Loader():

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def create(config_dict):
        return Loader(config_dict['data_dir'])


    def load_subject(self):
        subject = Subject(self.data_path)

        return subject

