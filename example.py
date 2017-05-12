from params import params_deepmedic
from src import dataset

# Get parameters for Dataset configuration
params_tuple = params_deepmedic.get_params_as_tuple()

# Create the desired dataset
BRATS_dataset = dataset.BRATS_dataset(params_tuple)

# Create the list of subjects to analayze
subject_list = BRATS_dataset.create_subjects()

# Load the image_segments desired
image_segments, label_segments = BRATS_dataset.load_image_segments(subject_list=subject_list)

# Create a data generator for model.fit_generator() function in Keras
data_generator = BRATS_dataset.data_generator(image_segments, label_segments)



