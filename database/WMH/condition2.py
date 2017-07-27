import os
from os.path import join
import nibabel as nib
import numpy as np
from src.utils.preprocessing import normalize_image
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
from skimage.morphology import binary_opening, binary_dilation, binary_closing
import copy

INPUT_DIR_LIST = ['/projects/neuro/WMH/Singapore','/projects/neuro/WMH/GE3T','/projects/neuro/WMH/Utrecht']
OUTPUT_DIR = '/projects/neuro/WMH/Utrecht'

def detect_outliers_down(array):
    bins,ths = np.histogram(array, bins=500)
    min_bins = bins[0]
    start_up = True
    for it_b, b in enumerate(bins[1:]):

        if b < min_bins:
            min_bins = b
            start_up = False
        else:
            if start_up:
                continue
            else:
                return ths[it_b]

def detect_outliers_up(array):
    bins,ths = np.histogram(array, bins=500)
    binsum_outliers = np.sum(bins[1:])*0.001
    for it_b, b in enumerate(bins[::-1]):
        if b > binsum_outliers:
            return ths[-it_b-1]

    # cum_sum = 0
    # for it_b, b in enumerate(bins[::-1]):
    #     cum_sum += b
    #     if cum_sum > binsum_outliers:
    #         return ths[-it_b-1]



# INPUT_DIR = INPUT_DIR_LIST[2]

for INPUT_DIR in INPUT_DIR_LIST:
    print(INPUT_DIR)
    subjects = os.listdir(INPUT_DIR)
    for subject in subjects:
        print(subject)
        # if subject not in ['109']:
        #     continue
        subject_path = join(INPUT_DIR,subject)
        modalities_path = join(subject_path,'pre')
        proxy = nib.load(join(modalities_path,'FLAIR.nii.gz'))
        image_array = np.asarray(proxy.dataobj)
        image_array = normalize_image(image_array)

        # th = detect_outliers_down(image_array)
        th = detect_outliers_up(image_array)

        seed = np.ones_like(image_array)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(image_array.flatten(),bins=500)
        # plt.show()
        # plt.close()

        image_otsu = copy.deepcopy(image_array)
        image_otsu[image_array < 0] = 0
        image_otsu[image_array > th] = 0
        thresh = threshold_otsu(image_otsu,nbins=500)

        seed[image_array > th] = 0
        seed[np.where(image_array < thresh)] = 0



        seed = np.array(seed, np.uint8)
        labels = label(seed)

        maxArea = 0
        for region in regionprops(labels):
            if region.area > maxArea:
                maxArea = region.area

        seed = remove_small_objects(labels, maxArea - 1)



        seed = ndi.binary_fill_holes(seed)
        seed = np.array(seed, np.uint8)

        # seed[seed > 0] = 1

        seed = binary_opening(seed, selem=np.ones((3, 3, 3)))

        seed = np.array(seed, np.uint8)

        labels = label(seed)

        maxArea = 0
        for region in regionprops(labels):
            if region.area > maxArea:
                maxArea = region.area

        seed_tmp = remove_small_objects(labels, maxArea - 1)
        
        img = nib.Nifti1Image(seed, proxy.affine)
        nib.save(img, join(modalities_path, 'mask_prova.nii.gz'))
        seed = ndi.binary_fill_holes(seed)
        seed = np.array(seed, np.uint8)
        # seed[seed > 0] = 1

        seed = binary_dilation(seed, selem=np.ones((3,3,3)))
        seed = binary_closing(seed, selem=np.ones((20, 1, 1)))

        seed = ndi.binary_fill_holes(seed)

        seed = np.array(seed, np.uint8)


        # #Mask 2
        # seed2 = np.ones_like(image_array)
        # seed2[np.where(image_array < 0.4)] = 0
        # seed2 = seed2 * seed
        #
        # img = nib.Nifti1Image(image_array, proxy.affine)
        # nib.save(img, join(modalities_path,'T1_norm.nii.gz'))
        #
        # img = nib.Nifti1Image(seed2, proxy.affine)
        # nib.save(img, join(modalities_path,'mask_morph2.nii.gz'))
        #
        #
        # import matplotlib.pyplot as plt
        #
        # plot_array = image_array.flatten() * seed.flatten()
        # plt.figure()
        # plt.hist(plot_array[plot_array != 0],500)
        # plt.show()
        # plt.close()


        img = nib.Nifti1Image(seed, proxy.affine)
        nib.save(img, join(modalities_path,'mask_morph.nii.gz'))

