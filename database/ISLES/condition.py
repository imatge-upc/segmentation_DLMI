import os
from os.path import join
import nibabel as nib
import numpy as np


INPUT_DIR = '/projects/neuro/ISLES2017/Training'
OUTPUT_DIR = '/projects/neuro/ISLES2017/Training'


subjects = os.listdir(INPUT_DIR)
for subject in subjects:
    output_path = join(OUTPUT_DIR, subject[9:])
    modalities_path = join(INPUT_DIR,subject)
    modalities = os.listdir(modalities_path)
    for modality in modalities:
        modality_path = join(modalities_path,modality)
        image_gt = modality.split(sep='.')[4]
        if 'MR' in image_gt:
            modality_name = image_gt.split('_')
            proxy = nib.load(join(modality_path,modality+'.nii'))
            image_array = np.asarray(proxy.dataobj)
            print(image_gt)
            print(image_array.shape)

    break