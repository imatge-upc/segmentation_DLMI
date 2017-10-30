import nibabel as nib
from glob import glob
import os
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from sklearn.metrics import recall_score, precision_score, accuracy_score
import numpy as np
import argparse

def dice_whole(y,y_pred):

    y_whole = y > 0
    y_pred_whole = y_pred > 0

    return (2. * np.sum(y_pred_whole * y_whole) + np.finfo(float).eps) / (np.sum(y_whole) + np.sum(y_pred_whole)+ np.finfo(float).eps)

def dice_core(y, y_pred):

    y_core = np.zeros(y.shape)
    y_core[np.where( (y==1) | (y==3) | (y==4) )[0]] = 1
    y_pred_core = np.zeros(y_pred.shape)
    y_pred_core[np.where( (y_pred==1) | (y_pred==3) | (y_pred==4) )[0] ] = 1

    return (2. * np.sum(y_pred_core*y_core) + np.finfo(float).eps) / (np.sum(y_core) + np.sum(y_pred_core)+ np.finfo(float).eps)

def dice_enhance(y,y_pred):
    y_enhance = np.zeros(y.shape)
    y_enhance[np.where(y == 4)[0]] = 1
    y_pred_enhance = np.zeros(y_pred.shape)
    y_pred_enhance[np.where( y_pred == 4)[0]] = 1

    return (2. * np.sum(y_pred_enhance * y_enhance) + np.finfo(float).eps ) / (np.sum(y_enhance) + np.sum(y_pred_enhance) + np.finfo(float).eps)

def remove_conn_components(pred_mask, num_cc):

    labels = label(pred_mask)

    if num_cc == 1:

        maxArea = 0
        for region in regionprops(labels):
            if region.area > maxArea:
                maxArea = region.area
                print(maxArea)

        mask = remove_small_objects(labels, maxArea - 1)

    else:
        mask = remove_small_objects(labels, 3000, connectivity=2)

    return mask



if __name__ == "__main__":

    """ CONSTANTS """
    DATA_PATH = '/projects/neuro/BRATS/BRATS2016_TESTING/'
    PREDICTIONS_PATH = '/work/acasamitjana/segmentation/BraTS/finale/20170815/LR_0.0005_v_net1_4_v_net2_v2/results_test_test'
    RESULTS_PATH = '/work/acasamitjana/segmentation/BraTS/finale/20170815/LR_0.0005_v_net1_4_v_net2_v2/results_test_test_morphology'
    NUM_CONNECTED_COMPONENTS = 1

    all_predictions = glob(os.path.join(PREDICTIONS_PATH, '**.nii.gz'))
    total_predictions = len(all_predictions)

    for ind, prediction_path in enumerate(all_predictions):

        # Get the subject name from the path
        filepath, filename = os.path.split(prediction_path)
        print(filename)


        # Get the predictions data
        pred_nii = nib.load(prediction_path)
        affine = pred_nii.get_affine()
        hdr = pred_nii.get_header()
        pred_img = pred_nii.get_data()
        pred_mask = np.asarray(pred_img > 0, np.uint8)


        # Remove small objects
        mask = remove_conn_components(pred_mask, NUM_CONNECTED_COMPONENTS)

        final_pred = pred_img * (pred_mask - (mask > 0))


        # Save the desired result ready to upload
        final_pred = nib.Nifti1Image(final_pred, affine, header=hdr)
        save_path = os.path.join(RESULTS_PATH, filename)
        nib.save(final_pred, save_path)


        print('{} / {} processed'.format(ind + 1, total_predictions))


    print('Done')