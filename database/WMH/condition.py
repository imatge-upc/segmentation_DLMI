import os
from os.path import join
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import laplace
from scipy.ndimage.morphology import binary_closing

INPUT_DIR_LIST = ['/projects/neuro/WMH/Utrecht','/projects/neuro/WMH/Singapore','/projects/neuro/WMH/GE3T']
OUTPUT_DIR = '/projects/neuro/WMH/Utrecht'



INPUT_DIR = INPUT_DIR_LIST[2]
print(INPUT_DIR)
subjects = os.listdir(INPUT_DIR)
for subject in subjects:
    print(subject)
    subject_path = join(INPUT_DIR,subject)
    modalities_path = join(subject_path,'pre')
    proxy = nib.load(join(modalities_path,'FLAIR.nii.gz'))
    image_array = np.asarray(proxy.dataobj)
    # filtered_image = laplace(image_array)
    # img = nib.Nifti1Image(filtered_image, proxy.affine)
    # nib.save(img, join(modalities_path,'laplaciona_FLAIR.nii.gz'))

    mask = np.ones_like(image_array)
    mask[np.where(image_array < 90)] = 0

    # img = nib.Nifti1Image(mask, proxy.affine)
    # nib.save(img, join(modalities_path,'mask.nii.gz'))

    struct_element_size = (20,20,20)
    mask_augmented = np.pad(mask,[(21,21),(21,21),(21,21)], 'constant', constant_values=(0,0))
    mask_augmented = binary_closing(mask_augmented,structure=np.ones(struct_element_size, dtype=bool)).astype(np.int)
    img = nib.Nifti1Image(mask_augmented[21:-21,21:-21,21:-21], proxy.affine)
    nib.save(img, join(modalities_path,'mask_closing.nii.gz'))

    # print('FLAIR')
    # print(image_array.mean())
    # print(image_array.std())
    # print(image_array.shape)
    # print('T1')
    # proxy = nib.load(join(modalities_path,'T1.nii.gz'))
    # image_array = np.asarray(proxy.dataobj)
    # print(image_array.mean())
    # print(image_array.std())
    # print(image_array.shape)
