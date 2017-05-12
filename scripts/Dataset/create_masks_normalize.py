import glob
import nibabel as nib
import numpy as np
from os.path import join, split

# Iterate over all NIFTI files
data_path = "/projects/neuro/BRATS/BRATS2016_TESTING"

data_path_glob = join(data_path, "data_test","*", "*.nii")
num_files = len(glob.glob(data_path_glob))
print "Number of files to be processed: ", num_files

for index, nifti_file_path in enumerate(glob.glob(data_path_glob)):
    # Parse path
    [file_path, file_name] = split(nifti_file_path)

    # Remove extension from file name
    file_name = ".".join(file_name.split(".")[:-1])

    # Load NIFTI file
    nii_file = nib.load(nifti_file_path)
    shape = nii_file.shape

    if shape[0]!=240:
        print file_name + str(shape)
    nii_file_data = nii_file.get_data()

    # Pad image
    pad = np.zeros(3, dtype=int)
    pad[0] = (32-shape[0]%32)%32
    pad[1] = (32-shape[1]%32)%32
    pad[2] = (32-shape[2]%32)%32

    nii_file_data = np.pad(nii_file_data, ((0, pad[0]), (0, pad[1]), (0, pad[2])), 'constant',
                           constant_values=(0, 0))

    # Create mask
    mask = np.zeros(nii_file_data.shape)
    valid_voxels = nii_file_data > 0
    mask[valid_voxels] = 1


    # Correct image to have zero mean and unit variance
    mean = np.sum(nii_file_data[valid_voxels]) / np.sum(mask)
    variance = np.sum(np.square(nii_file_data[valid_voxels] - mean)) / (np.sum(mask) - 1)
    std = np.sqrt(variance)
    corrected_data = (nii_file_data - mean) / std

    # Save mask
    mask_filename = join(file_path, file_name + '_mask.nii.gz')
    nib.save(nib.Nifti1Image(mask, nii_file.affine, nii_file.header), mask_filename)

    # Save corrected brain
    corrected_filename = join(file_path, file_name + '_corrected.nii.gz')
    nib.save(nib.Nifti1Image(corrected_data, nii_file.affine, nii_file.header), corrected_filename)

    # Print
    print index + 1, " files processed of ", num_files