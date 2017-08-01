import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from skimage import io
from os.path import join, normpath, basename
from glob import glob

INPUT_DIR = '/Users/acasamitjana/PhD/MICCAI17/iSeg/results_test'
OUTPUT_DIR = "/Users/acasamitjana/PhD/MICCAI17/iSeg/results_test/itksnap"


images = glob(join(INPUT_DIR,'*.nii.gz'))
for image in images:
    subject = basename(normpath(image))

    print(subject)

    proxy = nib.load(image)
    array = np.asarray(proxy.dataobj).astype('uint8')
    print(array.shape)
    sitk_image = sitk.GetImageFromArray(array, isVector=False)
    sitk.WriteImage(sitk_image,join(OUTPUT_DIR,subject[:-7]+".img") )
    # io.imsave(join(OUTPUT_DIR,subject[:-7]+'.img'),array,plugin='simpleitk')

    # img = nib.Nifti1Pair(array,proxy.affine, proxy.header)
    # nib.nifti1.save(img, join(OUTPUT_DIR,image[:-7]+'.img'))