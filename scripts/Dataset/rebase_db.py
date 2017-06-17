import nibabel as nib
from glob import glob
import os
import shutil


def function_to_filter(match_string, unmatched_string=None):
    # This function filters all all cases from 'iterable_string' that match all string in 'match_string'.
    # It is useful for the case of filtering all subdirectories that contain a certaing MRI modality +
    # a certain file extension
    if not isinstance(match_string, list):
        match_string = [match_string]

    if unmatched_string is None:
        def embeded_function(iterable_string):
            return True if all(word in iterable_string for word in match_string) else False
    else:
        if not isinstance(unmatched_string, list):
            unmatched_string = [unmatched_string]

        def embeded_function(iterable_string):
            return True if all(word in iterable_string for word in match_string) and not any(
                word in iterable_string for word in unmatched_string)else False

    return embeded_function


""" CONSTANTS """

DATA_PATH = '/projects/neuro/BRATS/BRATS2017_old/HGG'
REBASE_PATH = '/projects/neuro/BRATS/BRATS2017_Training/HGG'

subject_list = os.listdir(os.path.join(DATA_PATH))

i = 0
print_threshold = 10
num_subjects = len(subject_list)
for subject in subject_list:
    i += 1
    subject_path = os.path.join(REBASE_PATH,subject)
    old_subject_path = os.path.join(DATA_PATH, subject)
    os.mkdir(subject_path)

    folder_path = filter(function_to_filter('T1.'), os.listdir(old_subject_path))[0]
    t1_path = filter(function_to_filter('corrected.nii'), os.listdir(os.path.join(old_subject_path,folder_path)))[0]
    shutil.copy(os.path.join(old_subject_path,folder_path,t1_path), os.path.join(subject_path, 'T1.nii.gz'))


    folder_path = filter(function_to_filter('T1c.'), os.listdir(old_subject_path))[0]
    t1c_path = filter(function_to_filter('corrected.nii'), os.listdir(os.path.join(old_subject_path,folder_path)))[0]
    shutil.copy(os.path.join(old_subject_path,folder_path,t1c_path), os.path.join(subject_path, 'T1c.nii.gz'))

    folder_path = filter(function_to_filter('T2.'), os.listdir(old_subject_path))[0]
    t2_path = filter(function_to_filter('corrected.nii'), os.listdir(os.path.join(old_subject_path,folder_path)))[0]
    shutil.copy(os.path.join(old_subject_path, folder_path, t2_path), os.path.join(subject_path, 'T2.nii.gz'))

    folder_path = filter(function_to_filter('Flair.'), os.listdir(old_subject_path))[0]
    flair_path = filter(function_to_filter('corrected.nii'), os.listdir(os.path.join(old_subject_path,folder_path)))[0]
    shutil.copy(os.path.join(old_subject_path, folder_path, flair_path), os.path.join(subject_path, 'FLAIR.nii.gz'))

    folder_path = filter(function_to_filter('OT.'), os.listdir(old_subject_path))[0]
    labels_path = filter(function_to_filter('.nii', unmatched_string=['_corrected', '_mask']
                                            ), os.listdir(os.path.join(old_subject_path,folder_path)))[0]
    data_proxy = nib.load(os.path.join(old_subject_path, folder_path, labels_path))
    nib.save(data_proxy,os.path.join(subject_path, 'GT.nii.gz'))

    folder_path = filter(function_to_filter('T1c.'), os.listdir(old_subject_path))[0]
    roi_mask_path = filter(function_to_filter('mask'), os.listdir(os.path.join(old_subject_path,folder_path)))[0]
    shutil.copy(os.path.join(old_subject_path, folder_path, roi_mask_path), os.path.join(subject_path, 'ROImask.nii.gz'))

    evolution =  1.0*i/num_subjects*100

    if evolution > print_threshold:
        print_threshold += 10
        print str(evolution) + '% completed...'