import os
from os.path import join
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import laplace
from scipy.ndimage.morphology import binary_closing

INPUT_DIR_LIST = ['/projects/neuro/WMH/Utrecht','/projects/neuro/WMH/Singapore']#['/projects/neuro/WMH/GE3T']#


for INPUT_DIR in INPUT_DIR_LIST:
    subjects = os.listdir(INPUT_DIR)
    for subject in subjects:
        print(subject)
        subject_path = join(INPUT_DIR,subject)
        modalities_path = join(subject_path,'pre')

        proxy = nib.load(join(modalities_path,'FLAIR.nii.gz'))
        affine = proxy.affine
        affine[:, 2] = affine[:, 2] / 4
        # affine[:, 0] = affine[:, 0] / 4


        image_array = np.asarray(proxy.dataobj)
        image_out = np.repeat(image_array, 4, axis=2)
        # image_out = np.repeat(image_out[4:-4], 4, axis=0)

        img = nib.Nifti1Image(image_out, affine)
        nib.save(img, join(modalities_path, 'FLAIR_resampled.nii.gz'))


        proxy = nib.load(join(modalities_path, 'T1.nii.gz'))
        image_array = np.asarray(proxy.dataobj)
        image_out = np.repeat(image_array, 4, axis=2)
        # image_out = np.repeat(image_out[4:-4], 4, axis=0)

        img = nib.Nifti1Image(image_out, affine)
        nib.save(img, join(modalities_path, 'T1_resampled.nii.gz'))

        proxy = nib.load(join(subject_path, 'wmh.nii.gz'))
        image_array = np.asarray(proxy.dataobj)
        image_out = np.repeat(image_array, 4, axis=2)
        # image_out = np.repeat(image_out[4:-4], 4, axis=0)

        img = nib.Nifti1Image(image_out, affine)
        nib.save(img, join(subject_path, 'wmh_resample.nii.gz'))

        proxy = nib.load(join(modalities_path, 'mask_closing.nii.gz'))
        image_array = np.asarray(proxy.dataobj)
        image_out = np.repeat(image_array, 4, axis=2)
        # image_out = np.repeat(image_out[4:-4], 4, axis=0)

        img = nib.Nifti1Image(image_out, affine)
        nib.save(img, join(modalities_path, 'mask_closing_resample.nii.gz'))