import os
from os.path import join
import nibabel as nib
import numpy as np
from src.utils.preprocessing import normalize_image

INPUT_DIR = '/projects/neuro/BRATS/Brats17TestingData'
OUTPUT_DIR = '/projects/neuro/BRATS/BRATS2017_Testing'




print(INPUT_DIR)
subjects = os.listdir(INPUT_DIR)
N = len(subjects)
n = 0
for subject in subjects:

    if subject == 'survival_evaluation.csv':
        continue

    subject_path_in = join(INPUT_DIR,subject)
    subject_path_out = join(OUTPUT_DIR,subject)
    if not os.path.exists(subject_path_out):
        os.makedirs(subject_path_out)
    else:
        print(subject + ': ' + str(n) + ' / ' + str(N) + ' processed')
        n += 1
        continue



    #T1ce & mask
    proxy = nib.load(join(subject_path_in,subject + '_t1ce.nii.gz'))
    image_array = np.asarray(proxy.dataobj)
    mask = image_array > 0
    mask = mask.astype('int')
    img = nib.Nifti1Image(mask, proxy.affine)
    nib.save(img, join(subject_path_out, 'ROImask.nii.gz'))
    image_array = normalize_image(image_array, mask=mask)
    img = nib.Nifti1Image(image_array, proxy.affine)
    nib.save(img, join(subject_path_out, 'T1c.nii.gz'))

    #T1
    proxy = nib.load(join(subject_path_in,subject + '_t1.nii.gz'))
    image_array = np.asarray(proxy.dataobj)
    image_array = normalize_image(image_array, mask=mask)
    img = nib.Nifti1Image(image_array, proxy.affine)
    nib.save(img, join(subject_path_out, 'T1.nii.gz'))

    #T2
    proxy = nib.load(join(subject_path_in,subject + '_t2.nii.gz'))
    image_array = np.asarray(proxy.dataobj)
    image_array = normalize_image(image_array, mask=mask)
    img = nib.Nifti1Image(image_array, proxy.affine)
    nib.save(img, join(subject_path_out, 'T2.nii.gz'))

    #FLAIR
    proxy = nib.load(join(subject_path_in,subject + '_flair.nii.gz'))
    image_array = np.asarray(proxy.dataobj)
    image_array = normalize_image(image_array, mask=mask)
    img = nib.Nifti1Image(image_array, proxy.affine)
    nib.save(img, join(subject_path_out, 'FLAIR.nii.gz'))

    print(subject + ': ' + str(n) + ' / ' + str(N) + ' processed')
    n += 1