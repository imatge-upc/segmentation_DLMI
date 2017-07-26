import os
from glob import glob
import SimpleITK as sitk
from skimage import io
from os.path import join, normpath, basename, exists
import nibabel as nib
import numpy as np
from shutil import copyfile
from scipy.ndimage.morphology import binary_closing

INPUT_DIR = '/projects/neuro/IBSR/IBSR_nifti_stripped'
OUTPUT_DIR = '/projects/neuro/IBSR'


subjects = os.listdir(INPUT_DIR)
print(subjects)
for subject_id in subjects:
    if not os.path.exists(join(OUTPUT_DIR,subject_id)):
        os.makedirs(join(OUTPUT_DIR,subject_id))

    print(subject_id)

    labels_proxy = nib.load(join(INPUT_DIR, subject_id,subject_id+'_segTRI_fill_ana.nii.gz'))
    labels_array = np.squeeze(np.asarray(labels_proxy.dataobj))

    print(labels_array.shape)
    mask = np.ones_like(labels_array)
    mask[np.where(labels_array == 90)] = 0

    copyfile(join(INPUT_DIR, subject_id,subject_id+'_ana_strip.nii.gz'), join(OUTPUT_DIR,subject_id,'T1.nii.gz'))
    copyfile(join(INPUT_DIR, subject_id, subject_id + '_segTRI_fill_ana.nii.gz'), join(OUTPUT_DIR, subject_id,'GT.nii.gz'))
    img = nib.Nifti1Image(mask, np.ones((4,4)))
    nib.save(img, join(OUTPUT_DIR, subject_id, 'mask.nii.gz'))


