import os
from glob import glob
from skimage import io
from os.path import join, normpath, basename, exists
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import laplace
from scipy.ndimage.morphology import binary_closing

INPUT_DIR = '/projects/neuro/iSeg-2017/Training'
OUTPUT_DIR = '/projects/neuro/WMH/Utrecht'


images = glob(join(INPUT_DIR,'*img'))
for image in images:
    print(image)
    image_array = io.imread(image,plugin='simpleitk')
    subject = basename(normpath(image))
    subject_name = subject.split(sep='-')[1]
    modality_name = subject.split(sep='-')[2][:-4]


    if not exists(join(INPUT_DIR,subject_name)):
        os.makedirs(join(INPUT_DIR,subject_name))

    if modality_name == 'T1':
        mask = np.ones_like(image_array)
        mask[np.where(image_array < 90)] = 0

        struct_element_size = (20, 20, 20)
        mask_augmented = np.pad(mask, [(21, 21), (21, 21), (21, 21)], 'constant', constant_values=(0, 0))
        mask_augmented = binary_closing(mask_augmented, structure=np.ones(struct_element_size, dtype=bool)).astype(np.int)
        img = nib.Nifti1Image(mask_augmented[21:-21, 21:-21, 21:-21], np.ones((4,4)))
        nib.save(img, join(INPUT_DIR,subject_name,'mask.nii.gz'))

    img = nib.Nifti1Image(image_array, np.ones((4,4)))
    nib.save(img, join(INPUT_DIR,subject_name,modality_name+'.nii.gz'))

