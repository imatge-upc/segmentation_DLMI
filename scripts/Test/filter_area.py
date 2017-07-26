import nibabel as nib
from glob import glob
import os
from skimage.morphology import remove_small_objects
from src.utils import io_utils
from sklearn.metrics import recall_score, precision_score, accuracy_score
import numpy as np
import argparse
import sys

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("-separator", help="Name separator", type=str)
    parser.add_argument("-pred", help="Prediction folder", type=str)
    parser.add_argument("-results", help="Results folder", type=str)

    arg = parser.parse_args(sys.argv[1:])
    # separator = arg.separator
    pred_folder = arg.pred
    res_folder = arg.results


    """ CONSTANTS """
    DATA_PATH = '/projects/neuro/BRATS/BRATS2016_TESTING/'
    PREDICTIONS_PATH = '/imatge/acasamitjana/work/BRATS/output/test/'+pred_folder
    RESULTS_PATH = '/imatge/acasamitjana/work/BRATS/output/test_morphology/'+res_folder
    # SEPARATOR = separator#'u_net_old'#'two_pathways_train_new'#


    all_predictions = glob(os.path.join(PREDICTIONS_PATH, '**.nii.gz'))
    total_predictions = len(all_predictions)
    # a_acc = 0
    # d_w_acc = 0
    # d_c_acc = 0
    # d_e_acc = 0
    # r_0_acc = 0
    # r_1_acc = 0
    # r_2_acc = 0
    # r_3_acc = 0
    # r_4_acc = 0
    # p_0_acc = 0
    # p_1_acc = 0
    # p_2_acc = 0
    # p_3_acc = 0
    # p_4_acc = 0

    # """ REDIRECT STDOUT TO FILE """
    # io_utils.redirect_stdout_to_file(filepath=os.path.join(RESULTS_PATH,'results.txt'))

    for ind, prediction_path in enumerate(all_predictions):

        # Get the subject name from the path
        filepath, filename = os.path.split(prediction_path)

        # Get the predictions data
        pred_nii = nib.load(prediction_path)
        affine = pred_nii.get_affine()
        hdr = pred_nii.get_header()
        pred_img = pred_nii.get_data()

        # Remove small objects
        pred_mask = pred_img > 0
        mask = remove_small_objects(pred_mask,800,connectivity=2)
        final_pred = pred_img * mask

        # Save the desired result ready to upload
        final_pred = nib.Nifti1Image(final_pred, affine, header=hdr)
        save_path = os.path.join(RESULTS_PATH, filename)
        nib.save(final_pred, save_path)


        print('{} / {} processed'.format(ind + 1, total_predictions))


    print('Done')






    # a = accuracy_score(ground_truth.flatten(),final_pred.flatten())
    # d_w = dice_whole(ground_truth.flatten(),final_pred.flatten())
    # d_c = dice_core(ground_truth.flatten(),final_pred.flatten())
    # d_e = dice_enhance(ground_truth.flatten(),final_pred.flatten())
    # r_0 = recall_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [0],pos_label=0)
    # r_1 = recall_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [1],pos_label=1)
    # r_2 = recall_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [2],pos_label=2)
    # r_3 = recall_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [3],pos_label=3)
    # r_4 = recall_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [4],pos_label=4)
    # p_0 = precision_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [0],pos_label=0)
    # p_1 = precision_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [1],pos_label=1)
    # p_2 = precision_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [2],pos_label=2)
    # p_3 = precision_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [3],pos_label=3)
    # p_4 = precision_score(ground_truth.flatten(),final_pred.flatten(),average = None,labels = [4],pos_label=4)

    # a_acc = a_acc + a/total_predictions
    # d_w_acc = d_w_acc + d_w/total_predictions
    # d_c_acc = d_c_acc + d_c/total_predictions
    # d_e_acc = d_e_acc + d_e/total_predictions
    # r_0_acc = r_0_acc + r_0/total_predictions
    # r_1_acc = r_1_acc + r_1/total_predictions
    # r_2_acc = r_2_acc + r_2/total_predictions
    # r_3_acc = r_3_acc + r_3/total_predictions
    # r_4_acc = r_4_acc + r_4/total_predictions
    # p_0_acc = p_0_acc + p_0/total_predictions
    # p_1_acc =p_1_acc + p_1/total_predictions
    # p_2_acc = p_2_acc + p_2/total_predictions
    # p_3_acc = p_3_acc + p_3/total_predictions
    # p_4_acc = p_4_acc + p_4/total_predictions


    # print('Subject: ' + filename)
    # print('Accuracy: ' + str(a))
    # print('Dice whole: ' + str(d_w))
    # print('Dice core: ' + str(d_c))
    # print('Dice enhance: ' + str(d_e))
    # print('Recall 0 : ' + str(r_0))
    # print('Recall 1 : ' + str(r_1))
    # print('Recall 2 : ' + str(r_2))
    # print('Recall 3 : ' + str(r_3))
    # print('Recall 4 : ' + str(r_4))
    # print('Precision 0: ' + str(p_0))
    # print('Precision 1: ' + str(p_1))
    # print('Precision 2: ' + str(p_2))
    # print('Precision 3: ' + str(p_3))
    # print('Precision 4: ' + str(p_4))
    #     print('')

    #
    # print('----------------------------------------------------')
    # print('Total predictions')
    # print('Accuracy: ' + str(a_acc))
    # print('Dice whole: ' + str(d_w_acc))
    # print('Dice core: ' + str(d_c_acc))
    # print('Dice enhance: ' + str(d_e_acc))
    # print('Recall 0 : ' + str(r_0_acc))
    # print('Recall 1 : ' + str(r_1_acc))
    # print('Recall 2 : ' + str(r_2_acc))
    # print('Recall 3 : ' + str(r_3_acc))
    # print('Recall 4 : ' + str(r_4_acc))
    # print('Precision 0: ' + str(p_0_acc))
    # print('Precision 1: ' + str(p_1_acc))
    # print('Precision 2: ' + str(p_2_acc))
    # print('Precision 3: ' + str(p_3_acc))
    # print('Precision 4: ' + str(p_4_acc))