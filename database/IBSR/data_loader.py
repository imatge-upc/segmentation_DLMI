import nibabel as nib
import numpy as np
import os
from os.path import join
from src.utils.preprocessing import rotate_axial_plane, normalize_image

to_int = lambda b: 1 if b else 0

class Subject:
    def __init__(self,
                 data_path,
                 id):

        self.data_path = data_path
        self._id = id

        self.subject_filepath = join(self.data_path, self.id)

        self.T1_FILE = join(self.subject_filepath,'T1.nii.gz')
        self.GT_FILE = join(self.subject_filepath,'GT.nii.gz')
        self.ROIMASK_FILE = join(self.subject_filepath,'mask.nii.gz')

        self.data_augmentation = False



    @property
    def id(self):
        return self._id

    def get_affine(self):
        return nib.load(self.GT_FILE).affine

    def load_channels(self, normalize=False):

        proxy = nib.load(self.T1_FILE)

        if self.data_augmentation:
            channels = rotate_axial_plane(np.asarray(proxy.dataobj))
        else:
            channels = np.asarray(proxy.dataobj)

        if normalize:
            channels = normalize_image(channels)



        return channels

    def load_labels(self):
        labels_proxy = nib.load(self.GT_FILE)

        if self.data_augmentation:
            labels = rotate_axial_plane(np.squeeze(np.asarray(labels_proxy.dataobj)))
        else:
            labels = np.squeeze(np.asarray(labels_proxy.dataobj))

        unique_labels = np.unique(labels)
        unique_labels = np.sort(unique_labels)
        for it_u_l, u_l in enumerate(unique_labels):
            labels[np.where(labels == u_l)] = it_u_l
        return labels


    def load_ROI_mask(self):

        roimask_proxy = nib.load(self.ROIMASK_FILE)
        if self.data_augmentation:
            mask = rotate_axial_plane(np.asarray(roimask_proxy.dataobj))
        else:
            mask = np.asarray(roimask_proxy.dataobj)

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

    def load_subject(self, id):
        subject = Subject(id, self.data_path)

        return subject


    def load_subjects(self):

        subject_list = []
        data_path = self.data_path
        if not isinstance(self.data_path, list):
            data_path = [data_path]

        for path in data_path:
            for subject in os.listdir(path):
                subject_list.append(Subject(path,subject))

        return subject_list

    def _get_value(self, value):
        '''
         Checks that the value is not a nan, and returns None if it is
        '''
        if value == 'nan':
            return None

        if isinstance(value, str):
            # if value.isnumeric():
            #     return int(value)

            # Check if it can be casted to a float...
            try:
                return float(value)
            except ValueError:
                pass

        return value