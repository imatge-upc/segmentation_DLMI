import numpy as np
import os
from os.path import join
from src.utils.preprocessing import one_hot_representation, padding3D, resize_image, scale_image, normalize_image
from src.utils.sampling import sampling_scheme_selection
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import copy

np.random.seed(42)
DEBUG = False
to_int = lambda b: 1 if b else 0


class Dataset(object):
    def __init__(self,
                 input_shape,
                 output_shape,
                 n_classes,
                 num_modalities):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_classes = n_classes

        self.num_modalities = num_modalities




    def get_subject(self,subject_list, id):
        return filter(lambda subject: subject.id == id, subject_list)[0]

    def get_affine(self,subject_list):
        subj = subject_list[0]
        return subj.get_affine()

    def set_subject_list(self, subject_list):
        self.subject_list = subject_list

    def data_generator_one(self, subject):
        image_batch = np.zeros((1,) + self.input_shape + (self.num_modalities,))

        image = subject.load_channels()
        for mod in range(image.shape[-1]):
            image_batch[0,:, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape, pad_value=image[0, 0, 0, mod])
        labels = resize_image(subject.load_labels(),self.input_shape, pad_value = 0)[np.newaxis,:]

        return (image_batch, one_hot_representation(labels, self.n_classes), subject.id)


class Dataset_train(Dataset):

    def __init__(self,
                 input_shape,
                 output_shape,
                 n_classes,
                 n_subepochs,
                 batch_size,

                 sampling_scheme,
                 sampling_weights,

                 n_subjects_per_epoch_train,
                 n_segments_per_epoch_train,
                 n_subjects_per_epoch_validation,
                 n_segments_per_epoch_validation,

                 train_size,
                 dev_size,

                 num_modalities,

                 data_augmentation_flag = False,
                 data_augmentation_planes = None,
                 class_weights = 'constant'):

        self.n_subepochs = n_subepochs
        self._class_weights = class_weights
        self.sampling_scheme = sampling_scheme_selection(sampling_scheme, n_classes)
        self.sampling_scheme.set_probabilities(sampling_weights)
        self.batch_size = batch_size

        self.n_segments_per_epoch_train = n_segments_per_epoch_train
        self.n_subjects_per_subepoch_train = n_subjects_per_epoch_train
        self.n_segments_per_epoch_validation = n_segments_per_epoch_validation
        self.n_subjects_per_subepoch_validation = n_subjects_per_epoch_validation

        self.train_size = train_size
        self.dev_size = dev_size

        print('Data_augmentation_flag: ' + str(data_augmentation_flag))
        self.data_augmentation_flag = data_augmentation_flag
        self.data_augmentation_planes = data_augmentation_planes

        super(Dataset_train,self).__init__(input_shape,
                                           output_shape,
                                           n_classes,
                                           num_modalities
                                           )

    def class_weights(self, subject_list, mask_bool ='ROI'):
        if self._class_weights == 'constant' or self._class_weights is None:
            return None
        elif self._class_weights == 'inverse_weights':
            return self.get_class_weights(subject_list, mask_bool=mask_bool)
        else:
            raise ValueError("Please, specify a correct weighting of the cost function")

    def split_train_val(self, subject_list):
        L = len(subject_list)

        np.random.seed(42)
        np.random.shuffle(subject_list)
        L_train = int(np.round(self.train_size * L))
        L_val = int(np.round((1 - self.train_size) * L - np.finfo(float).eps))

        if L_val == 0:
            return subject_list, subject_list
        else:
            return subject_list[:L_train], subject_list[L_train:L_train + L_val]

    def _get_subjects_subepoch(self, subject_list, n_subjects):
        # Choose from which subjects take the data, taking into account that it can be greater than the number of subjects present in our dataset
        total_n_subjects = len(subject_list)

        if n_subjects > total_n_subjects:
            subjects_chosen = subject_list
            len_chosen_subjects = total_n_subjects
            while n_subjects - len_chosen_subjects < total_n_subjects:
                subjects_chosen.extend(subject_list)
                len_chosen_subjects += total_n_subjects
            subjects_chosen.extend(subject_list[:n_subjects - len_chosen_subjects])
        else:
            subjects_chosen = subject_list[:n_subjects]

        return subjects_chosen


    def sampling_parameters(self,
                            n_subjects,
                            n_segments,
                            prob_sampling_samples
                            ):

        # This function implements positive/negative sampling of the tumor/nontumor regions
        n_categories = len(prob_sampling_samples)

        # Choose the segments (spread evenly to all subjects) to be loaded into GPU. If n_segments is not a multiple of
        # n_subjects, then the first subjects will have more segments until we reach the desired number of segments
        n_segments_per_category= np.ones(n_categories, dtype='int32') * np.floor(n_segments / n_categories)
        np.random.seed(42)
        index_fill_categories = np.random.choice(n_categories, size=int(n_segments % n_categories), replace=True,p=prob_sampling_samples)
        for index_cat in index_fill_categories:
            n_segments_per_category[index_cat] += 1

        # If spliting exact number of segments between classes, take one more of the positive class
        n_segments_distribution = np.zeros((n_subjects, n_categories),dtype = "int32")
        for index_cat in range(n_categories):
            n_segments_distribution[:,index_cat] = np.floor( n_segments_per_category[index_cat] / n_subjects )
            np.random.seed(42)
            n_segments_distribution[np.random.randint(0, int(n_subjects), size=int(n_segments_per_category[index_cat]) % int(n_subjects), dtype="int32"),
                                    int(index_cat)] += 1

        return n_segments_distribution


    def load_image_segments(self,
                            subject_list,
                            n_subjects,
                            n_segments,
                            samplingInstance,
                            normalize_bool = True
                            ):

        subject_list = self._get_subjects_subepoch(subject_list, n_subjects)
        prob_sampling_samples = samplingInstance.get_probabilities()
        n_segments_distribution = self.sampling_parameters(n_subjects,
                                                           n_segments,
                                                           prob_sampling_samples
                                                           )


        image_segments = np.zeros((np.sum(n_segments_distribution),) + tuple(self.input_shape) + (self.num_modalities,),dtype=np.float32)
        labels_patch_segments = np.zeros((np.sum(n_segments_distribution),) + tuple(self.output_shape),dtype=np.int32)

        # Centered in the input image if input_shape - output_shape is even. Otherwise it will be one voxel to the right
        gt_patch_width = (np.asarray(self.input_shape, dtype=int) - np.asarray(self.output_shape, dtype=int)) / 2
        gt_patch_width = gt_patch_width.astype(int)

        cur_segment = 0
        for index, subject in enumerate(subject_list):
            image_channels = subject.load_channels(normalize=normalize_bool) #4D array [dim0,dim1,dim2,channels]
            image_labels = subject.load_labels() #3D array [dim0,dim1,dim2]
            roi_mask = subject.load_ROI_mask()

            # image_channels, image_labels, roi_mask = samplingInstance.get_proper_images(self.input_shape,
            #                                                                             image_channels,
            #                                                                             image_labels,
            #                                                                             roi_mask)


            mask = samplingInstance.get_weighted_mask(self.input_shape, image_labels.shape,
                                                      ROI_mask=roi_mask,labels_mask=image_labels)

            central_voxels_coordinates, segments_coordinates = self._get_segments_coordinates(self.input_shape,
                                                                                              mask,
                                                                                              n_segments_distribution[index,:]
                                                                                              )

            if DEBUG:
                print('Subject: ' + subject.id)
                print('Class distribution of subject: ' + str(np.bincount(image_labels.reshape(-1, ).astype(int), minlength=self.n_classes)))


            for index_segment in range(central_voxels_coordinates.shape[1]):
                coor = segments_coordinates[:, index_segment, :]
                image_segments[cur_segment] = image_channels[coor[0, 0]:coor[0, 1] + 1,
                                              coor[1, 0]:coor[1, 1] + 1,
                                              coor[2, 0]:coor[2, 1] + 1, :]
                labels_whole_segment = image_labels[coor[0, 0]:coor[0, 1] + 1, coor[1, 0]:coor[1, 1] + 1,
                                       coor[2, 0]:coor[2, 1] + 1]

                # print(gt_patch_width)
                # print(self.output_shape)
                # print(labels_whole_segment.shape)
                # print(cur_segment)
                # print(labels_patch_segments.shape)
                labels_patch_segments[cur_segment] = labels_whole_segment[
                                                     gt_patch_width[0]:self.output_shape[0] + gt_patch_width[0],
                                                     gt_patch_width[1]:self.output_shape[1] + gt_patch_width[1],
                                                     gt_patch_width[2]:self.output_shape[2] + gt_patch_width[2]]
                cur_segment += 1


        image_segments, labels_patch_segments = self.data_shuffle(image_segments, labels_patch_segments)


        return (image_segments, labels_patch_segments)

    def _get_segments_coordinates(self,
                                  segment_dimensions,
                                  weighted_mask,
                                  n_segments_distribution
                                  ):

        #Here we get the dimesions from the central voxel to the segment boundaries in each axis.
        #If dimension is even (dim%2==0), then the central voxel will be close to the left than to the right
        #                                 (i.e: dim=10, [4, central_voxel, 5]
        #If dimension is odd, then the central voxel is just in the middel
        coord_central_voxels = np.empty((len(segment_dimensions), 0), dtype='int16')


        half_segment_dimensions = np.zeros((len(segment_dimensions), 2), dtype='int16')
        for index, dim in enumerate(segment_dimensions):
            if dim % 2 == 0:
                half_segment_dimensions[index, :] = [dim / 2 - 1, dim / 2]
            else:
                half_segment_dimensions[index, :] = [np.floor(dim / 2)] * 2

        for index_cat in range(n_segments_distribution.shape[0]):
            n_segments = n_segments_distribution[index_cat]
            mask = weighted_mask[index_cat,:]
            indices_central_voxels_flattened = np.random.choice(mask.size,
                                                                size=int(n_segments),
                                                                replace = False,
                                                                p=mask.flatten())

            if DEBUG: print("Central voxels chosen in class: " + str(index_cat) + str(indices_central_voxels_flattened))

            coord_central_voxels = np.concatenate((coord_central_voxels,
                                                   np.asarray(np.unravel_index(indices_central_voxels_flattened,
                                                                        mask.shape))), axis=1)

        coord_segments = np.zeros(list(coord_central_voxels.shape) + [2], dtype='int32')
        coord_segments[:, :, 0] = coord_central_voxels - half_segment_dimensions[:, np.newaxis, 0]
        coord_segments[:, :, 1] = coord_central_voxels + half_segment_dimensions[:, np.newaxis, 1]

        return (coord_central_voxels, coord_segments)


    def data_generator(self, subject_list, batch_size = None, mode = 'train', data_format = 'channels_last',
                       normalize_bool = True):

        batch_size = batch_size if batch_size is not None else self.batch_size
        if mode == 'train':
            n_segments_subepoch = int(np.ceil(self.n_segments_per_epoch_train / self.n_subepochs))
            subject_list = self.data_augmentation(subject_list)
            n_subjects = len(subject_list) if self.n_subjects_per_subepoch_train is None else self.n_subjects_per_subepoch_train

        elif mode == 'validation':
            n_segments_subepoch = int(np.ceil(self.n_segments_per_epoch_validation / self.n_subepochs))
            n_subjects = len(subject_list) if self.n_subjects_per_subepoch_validation is None else self.n_subjects_per_subepoch_validation
        else:
            raise ValueError('Please, specify a valid generator mode [train, validation]')



        while True:

            for subepoch in range(self.n_subepochs):
                np.random.seed(42)
                np.random.shuffle(subject_list)
                image_segments, label_segments = self.load_image_segments(subject_list,
                                                                          n_subjects,
                                                                          n_segments_subepoch,
                                                                          self.sampling_scheme,
                                                                          normalize_bool = normalize_bool
                                                                          )
                s = np.arange(image_segments.shape[0])
                np.random.seed(42)
                np.random.shuffle(s)
                image_segments=image_segments[s,:,:,:,:]
                label_segments = label_segments[s,:,:,:]

                n_iter = int(np.ceil(len(image_segments) / float(batch_size)))
                for i in range(n_iter):
                    try:
                        features = image_segments[i * batch_size:(i + 1) * batch_size, :, :, :, :]
                        labels = one_hot_representation(label_segments[i * batch_size:(i + 1) * batch_size, :, :, :],
                                                        self.n_classes)
                    except:

                        features = image_segments[i * batch_size:, :, :, :, :]
                        labels = one_hot_representation(label_segments[i * batch_size:, :, :, :], self.n_classes)

                    if data_format == 'channels_first':
                        features = np.transpose(features, (0, 4, 1, 2, 3))
                        labels = np.transpose(labels, (0, 4, 1, 2, 3))

                    yield features, labels

    def data_generator_full(self, subject_list, mode='train', normalize_bool = True, mask = True):
        if mode == 'train':
            subject_list = self.data_augmentation(subject_list)

        it_subject_batch = 0
        image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
        labels_batch = np.zeros((self.batch_size,) + self.input_shape)
        mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

        while True:
            for subject in subject_list:
                image = subject.load_channels(normalize = normalize_bool)
                for mod in range(image.shape[-1]):
                    image_batch[it_subject_batch, :, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape,
                                                                               pad_value=image[0, 0, 0, mod])
                mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_ROI_mask(), self.input_shape, pad_value=0)
                labels_batch[it_subject_batch, :, :, :] = resize_image(subject.load_labels(), self.input_shape, pad_value=0)

                if it_subject_batch + 1 == self.batch_size:
                    it_subject_batch = 0
                    if mask:
                        yield ([image_batch, mask], one_hot_representation(labels_batch, self.n_classes))
                    else:
                        yield (image_batch,one_hot_representation(labels_batch, self.n_classes) )

                else:
                    it_subject_batch += 1

    def get_class_distribution(self, subject_list):

        class_frequencies = np.zeros(self.n_classes)

        for subj in subject_list:
            labels = subj.load_labels()
            mask = subj.load_ROI_mask()
            class_frequencies += np.bincount(labels.flatten().astype('int'), weights=mask.flatten(),
                                             minlength=self.n_classes)

        return class_frequencies

    def get_class_weights(self,subject_list, mask_bool = True):

        class_frequencies = np.zeros(self.n_classes)

        for subj in subject_list:
            labels = subj.load_labels()
            if mask_bool == 'ROI':
                mask = subj.load_ROI_mask()
                class_frequencies += np.bincount(labels.flatten().astype('int'), weights=mask.flatten().astype('int'),
                                                 minlength=self.n_classes)
            elif mask_bool == 'labels':
                mask = np.zeros_like(labels)
                mask[labels > 0] = 1
                # print(np.bincount(labels.flatten().astype('int'), weights=mask.flatten().astype('int'),
                #                                  minlength=self.n_classes))
                class_frequencies += np.bincount(labels.flatten().astype('int'), weights=mask.flatten().astype('int'),
                                                 minlength=self.n_classes+1)[1:]
            else :
                class_frequencies += np.bincount(labels.flatten().astype('int'),
                                                 minlength=self.n_classes)

        class_frequencies = class_frequencies / np.sum(class_frequencies)
        class_weight = np.sort(class_frequencies)[int(np.ceil(1.0*self.n_classes/2))] / class_frequencies
        class_weight[np.where(class_frequencies == 0)[0]] = 0 #avoid infinit weight

        return class_weight

    def data_augmentation(self, subject_list):

        if self.data_augmentation_flag:
            for d in self.data_augmentation_planes:
                subject_list_augmentation = copy.deepcopy(subject_list)
                if d == 'saggital-plane':
                    for sbj in subject_list_augmentation:
                        sbj.data_augmentation = 0
                        sbj.data_augmentation_flag = True

                if d == 'coronal-plane':
                    for sbj in subject_list_augmentation:
                        sbj.data_augmentation = 1
                        sbj.data_augmentation_flag = True

                if d == 'axial-plane':
                    for sbj in subject_list_augmentation:
                        sbj.data_augmentation = 2
                        sbj.data_augmentation_flag = True

                subject_list += subject_list_augmentation



        return subject_list

    @staticmethod
    def data_shuffle(image,labels):
        index_shuffle = np.arange(image.shape[0])
        np.random.shuffle(index_shuffle)

        return (image[index_shuffle], labels[index_shuffle])


class Dataset_test(Dataset):

    def __init__(self,
                 input_shape,
                 output_shape,
                 n_classes,
                 batch_size,
                 num_modalities
                 ):

        self.batch_size = batch_size

        super(Dataset_test,self).__init__(input_shape,
                                          output_shape,
                                          n_classes,
                                          num_modalities
                                          )

    def data_generator_inference(self, subject_list, normalize_bool = False):

        it_subject_batch = 0
        image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
        mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

        while True:

            for subject in subject_list:
                image = subject.load_channels(normalize=normalize_bool)
                for mod in range(image.shape[-1]):
                    image_batch[it_subject_batch,:, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape,
                                                                              pad_value=image[0, 0, 0, mod])

                mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_ROI_mask(), self.input_shape,
                                                                  pad_value=0)
                if it_subject_batch+1==self.batch_size:
                    it_subject_batch = 0
                    yield [image_batch, mask]
                else:
                    it_subject_batch+=1


class Dataset_Brats(Dataset_train):

    def __init__(self,
                 input_shape,
                 output_shape,
                 n_classes,
                 n_subepochs,
                 batch_size,

                 sampling_scheme,
                 sampling_weights,

                 n_subjects_per_epoch_train,
                 n_segments_per_epoch_train,
                 n_subjects_per_epoch_validation,
                 n_segments_per_epoch_validation,

                 train_size,
                 dev_size,

                 num_modalities,

                 data_augmentation_flag =False,
                 class_weights = 'constant'):

        super(Dataset_Brats,self).__init__(input_shape,
                                           output_shape,
                                           n_classes,
                                           n_subepochs,
                                            batch_size,

                                            sampling_scheme,
                                            sampling_weights,

                                            n_subjects_per_epoch_train,
                                            n_segments_per_epoch_train,
                                            n_subjects_per_epoch_validation,
                                            n_segments_per_epoch_validation,

                                            train_size,
                                            dev_size,

                                            num_modalities,

                                            data_augmentation_flag,
                                            class_weights
                                            )


    def data_generator_BraTS_survival(self, subject_list, mode='train', normalize_bool=True,
                                           sample_weights_bool=False):

        if mode == 'train':
            subject_list = self.data_augmentation(subject_list)

        it_subject_batch = 0
        image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
        labels_survival_batch = np.zeros((self.batch_size,))
        boolean_lables_survival = np.zeros((self.batch_size,))
        mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
        tumor_mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

        while True:
            for subject in subject_list:
                image = subject.load_channels(normalize=normalize_bool)
                for mod in range(image.shape[-1]):
                    image_batch[it_subject_batch, :, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape,
                                                                               pad_value=image[0, 0, 0, mod])
                mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_ROI_mask(), self.input_shape,
                                                                  pad_value=0)
                tumor_mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_tumor_mask(), self.input_shape,
                                                                        pad_value=0)

                survival = subject.load_survival()
                if survival is None:
                    labels_survival_batch[it_subject_batch] = 0
                    boolean_lables_survival[it_subject_batch] = 0
                else:
                    labels_survival_batch[it_subject_batch] = survival
                    boolean_lables_survival[it_subject_batch] = 1

                if it_subject_batch + 1 == self.batch_size:

                    if sample_weights_bool:

                        yield ([image_batch, mask, boolean_lables_survival],
                               labels_survival_batch)
                    else:
                        yield ([image_batch, mask, boolean_lables_survival],
                               labels_survival_batch)
                    it_subject_batch = 0
                    image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
                    labels_survival_batch = np.zeros((self.batch_size,))
                    boolean_lables_survival = np.zeros((self.batch_size,))
                    mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
                    tumor_mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

                else:
                    it_subject_batch += 1

    def data_generator_BraTS_mask(self, subject_list, mode='train', normalize_bool=True, sample_weights_bool=False,
                                  class_weights=None):
        if mode == 'train':
            subject_list = self.data_augmentation(subject_list)

        it_subject_batch = 0
        image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
        labels_batch = np.zeros((self.batch_size,) + self.input_shape)
        mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

        while True:
            np.random.shuffle(subject_list)
            for subject in subject_list:
                image = subject.load_channels(normalize=normalize_bool)
                for mod in range(image.shape[-1]):
                    image_batch[it_subject_batch, :, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape,
                                                                               pad_value=image[0, 0, 0, mod])
                mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_ROI_mask(), self.input_shape,
                                                                  pad_value=0)
                labels_batch[it_subject_batch, :, :, :] = resize_image(subject.load_tumor_mask(), self.input_shape,
                                                                       pad_value=0)

                if it_subject_batch + 1 == self.batch_size:
                    it_subject_batch = 0
                    if sample_weights_bool:
                        labels_batch_resized = np.reshape(labels_batch, (self.batch_size, -1))
                        sample_weights = np.zeros_like(labels_batch_resized)
                        for k, v in class_weights.items():
                            sample_weights[np.where(labels_batch_resized == k)] = v

                        yield ([image_batch, mask], one_hot_representation(labels_batch_resized, 2),
                               sample_weights)
                    else:
                        yield ([image_batch, mask], one_hot_representation(labels_batch, 2))

                    image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
                    labels_batch = np.zeros((self.batch_size,) + self.input_shape)
                    mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
                else:
                    it_subject_batch += 1

    def data_generator_BraTS_seg(self, subject_list, mode='train', normalize_bool=True, sample_weights_bool=False,
                                 class_weights=None):
        if mode == 'train':
            subject_list = self.data_augmentation(subject_list)

        it_subject_batch = 0
        image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
        labels_seg_batch = np.zeros((self.batch_size,) + self.input_shape)
        tumor_mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
        mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

        while True:
            for subject in subject_list:
                image = subject.load_channels(normalize=normalize_bool)
                for mod in range(image.shape[-1]):
                    image_batch[it_subject_batch, :, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape,
                                                                               pad_value=image[0, 0, 0, mod])
                tumor_mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_tumor_mask(), self.input_shape,
                                                                        pad_value=0)
                mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_ROI_mask(), self.input_shape,
                                                                  pad_value=0)

                labels_seg_batch[it_subject_batch, :, :, :] = resize_image(subject.load_labels(), self.input_shape,
                                                                           pad_value=0)

                if it_subject_batch + 1 == self.batch_size:
                    it_subject_batch = 0
                    if sample_weights_bool:
                        yield ([image_batch, mask, tumor_mask],
                               np.concatenate((
                                   one_hot_representation(labels_seg_batch, 4, class_weights=class_weights)[:, :, :, :,
                                   1:4],
                                   one_hot_representation(labels_seg_batch, 4, class_weights=class_weights)[:, :, :, :,
                                   5:],)
                                   , axis=4)
                               )
                    else:
                        yield (
                        [image_batch, mask, tumor_mask], one_hot_representation(labels_seg_batch, 4)[:, :, :, :, 1:])

                    image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
                    labels_seg_batch = np.zeros((self.batch_size,) + self.input_shape)
                    tumor_mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
                    mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
                else:
                    it_subject_batch += 1

    def data_generator_BraTS_mask_seg(self, subject_list, mode='train', normalize_bool=True,
                                      sample_weights_bool=False, class_weights=None, n_iterations=None
                                      ):
        if mode == 'train':
            subject_list = self.data_augmentation(subject_list)

        n_iterations = n_iterations if n_iterations is not None else len(subject_list)

        it_subject_batch = 0
        image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
        labels_mask_batch = np.zeros((self.batch_size,) + self.input_shape)
        labels_seg_batch = np.zeros((self.batch_size,) + self.input_shape)
        mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
        tumor_mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

        while True:
            np.random.shuffle(subject_list)
            n_sbj = 0
            for subject in subject_list:

                image = subject.load_channels(normalize=normalize_bool)
                for mod in range(image.shape[-1]):
                    image_batch[it_subject_batch, :, :, :, mod] = resize_image(image[:, :, :, mod], self.input_shape,
                                                                               pad_value=image[0, 0, 0, mod])

                tumor_mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_tumor_mask(), self.input_shape,
                                                                        pad_value=0)
                mask[it_subject_batch, :, :, :, 0] = resize_image(subject.load_ROI_mask(), self.input_shape,
                                                                  pad_value=0)

                labels_mask_batch[it_subject_batch, :, :, :] = resize_image(subject.load_tumor_mask(), self.input_shape,
                                                                            pad_value=0)
                labels_seg_batch[it_subject_batch, :, :, :] = resize_image(subject.load_labels(), self.input_shape,
                                                                           pad_value=0)

                if it_subject_batch + 1 == self.batch_size:
                    if mode == 'test':
                        yield ([image_batch, mask, mask],
                               [one_hot_representation(labels_mask_batch, 2),
                                one_hot_representation(labels_seg_batch, 4),
                                ], subject)
                    else:

                        if sample_weights_bool:
                            # sample_weights = np.zeros(labels_seg_batch)
                            # for k, v in class_weights.items():
                            #     sample_weights[np.where(labels_batch_resized == k)] = v

                            yield ([image_batch, mask, tumor_mask],
                                   [one_hot_representation(labels_mask_batch, 2),
                                    np.concatenate((
                                        one_hot_representation(labels_seg_batch, 4, class_weights=class_weights)[:, :,
                                        :, :, 1:4],
                                        one_hot_representation(labels_seg_batch, 4, class_weights=class_weights)[:, :,
                                        :, :, 5:],)
                                        , axis=4),
                                    ])
                        else:
                            yield ([image_batch, mask, tumor_mask],
                                   [one_hot_representation(labels_mask_batch, 2),
                                    one_hot_representation(labels_seg_batch, 4)[:, :, :, :, 1:],
                                    ])
                    it_subject_batch = 0
                    image_batch = np.zeros((self.batch_size,) + self.input_shape + (self.num_modalities,))
                    labels_mask_batch = np.zeros((self.batch_size,) + self.input_shape)
                    labels_seg_batch = np.zeros((self.batch_size,) + self.input_shape)
                    mask = np.zeros((self.batch_size,) + self.input_shape + (1,))
                    tumor_mask = np.zeros((self.batch_size,) + self.input_shape + (1,))

                else:
                    it_subject_batch += 1

                n_sbj += 1
                if n_sbj >= n_iterations:
                    break
