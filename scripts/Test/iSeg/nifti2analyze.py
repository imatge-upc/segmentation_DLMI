import nibabel as nib
import numpy as np
import os
from skimage import io
from os.path import join, normpath, basename
from glob import glob

INPUT_DIR = '/work/acasamitjana/segmentation/iSeg/20170728/VNet_patches/LR_0.0005_full_DA_shortcutTrue/results_test'
OUTPUT_DIR = '/work/acasamitjana/segmentation/iSeg/20170728/VNet_patches/LR_0.0005_full_DA_shortcutTrue/results_test/analyze'


images = glob(join(INPUT_DIR,'*.nii.gz'))
for image in images:
    subject = basename(normpath(image))

    print(subject)

    proxy = nib.load(image)
    array = np.asarray(proxy.dataobj).astype('uint8')
    print(array.shape)
    io.imsave(join(OUTPUT_DIR,subject[:-7]+'.img'),array,plugin='simpleitk')

    # img = nib.Nifti1Pair(array,proxy.affine, proxy.header)
    # nib.nifti1.save(img, join(OUTPUT_DIR,image[:-7]+'.img'))