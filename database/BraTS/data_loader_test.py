import nibabel as nib
import numpy as np
import os
from os.path import join
from src.utils.preprocessing import flip_plane, normalize_image
import csv


to_int = lambda b: 1 if b else 0
SURVIVAL_PATH = '/projects/neuro/BRATS/BRATS2017_Training/survival_data.csv'

class Subject:
    def __init__(self,
                 data_path,
                 id):

        self.data_path = data_path
        self._id = id

        self.subject_filepath = join(self.data_path, self.id)

        self.T1_FILE = join(self.subject_filepath,'T1.nii.gz')
        self.FLAIR_FILE = join(self.subject_filepath,'FLAIR.nii.gz')
        self.T1_FILE = join(self.subject_filepath,'T1.nii.gz')
        self.T1c_FILE = join(self.subject_filepath,'T1c.nii.gz')
        self.T2_FILE = join(self.subject_filepath,'T2.nii.gz')

        self.GT_FILE = join(self.subject_filepath,'ROImask.nii.gz')
        self.ROIMASK_FILE = join(self.subject_filepath,'ROImask.nii.gz')
        self.TUMORMASK_FILE = join(self.subject_filepath, 'ROImask.nii.gz')

        self.data_augmentation_flag = False
        self.data_augmentation = 0



    @property
    def id(self):
        return self._id

    def get_affine(self):
        return nib.load(self.GT_FILE).affine

    def load_channels(self, normalize=False):

        modalities = []
        modalities.append(nib.load(self.FLAIR_FILE))
        modalities.append(nib.load(self.T1_FILE))
        modalities.append(nib.load(self.T1c_FILE))
        modalities.append(nib.load(self.T2_FILE))

        channels = np.zeros(modalities[0].shape + (4,), dtype=np.float32)

        if normalize:
            mask = self.load_ROI_mask()

        for index_mod, mod in enumerate(modalities):
            if self.data_augmentation_flag:
                channels[:, :, :, index_mod] = flip_plane(np.asarray(mod.dataobj), plane=self.data_augmentation)
            else:
                channels[:, :, :, index_mod] = np.asarray(mod.dataobj)

            if normalize:
                channels[:, :, :, index_mod] = normalize_image(channels[:, :, :, index_mod], mask=mask)

        return channels

    def load_labels(self):
        labels_proxy = nib.load(self.GT_FILE)

        if self.data_augmentation_flag:
            labels = flip_plane(np.squeeze(np.asarray(labels_proxy.dataobj)),  plane=self.data_augmentation)
        else:
            labels = np.squeeze(np.asarray(labels_proxy.dataobj))

        return labels

    def load_ROI_mask(self):

        roimask_proxy = nib.load(self.ROIMASK_FILE)
        if self.data_augmentation_flag:
            mask = flip_plane(np.asarray(roimask_proxy.dataobj), plane=self.data_augmentation)
        else:
            mask = np.asarray(roimask_proxy.dataobj)

        return mask.astype('bool')

    def load_tumor_mask(self):
        tumormask_proxy = nib.load(self.TUMORMASK_FILE)
        if self.data_augmentation_flag:
            mask = flip_plane(np.asarray(tumormask_proxy.dataobj), plane=self.data_augmentation)
        else:
            mask = np.asarray(tumormask_proxy.dataobj)

        return mask.astype('bool')

    def load_survival(self):

        with open(SURVIVAL_PATH, newline='') as clinical_file:
            reader = csv.DictReader(clinical_file)
            for row in reader:
                if row['Brats17ID'] == self._id:
                    return self._normalize_surival(float(row['Survival']))
        return None

    def _normalize_surival(self,x):
        if hasattr(self,'survival_mean') and hasattr(self,'survival_std'):
            return (x-self.survival_mean) / self.survival_std
        else:
            return x

    def get_subject_shape(self):

        proxy = nib.load(self.T1_FILE)
        return proxy.shape

