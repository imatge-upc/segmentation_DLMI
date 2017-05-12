import numpy as np
import os
from os.path import join
from src.helpers.subject import Subject, Subject_test
from src.helpers.preprocessing_utils import one_hot_representation, padding3D
from src.helpers.sampling import sampling_scheme_selection

np.random.seed(42)
DEBUG = False
to_int = lambda b: 1 if b else 0


class Dataset(object):
    def __init__(self,
                 input_shape,
                 output_shape,
                 data_path,
                 n_classes,

                 booleanFLAIR,
                 booleanT1,
                 booleanT1c,
                 booleanT2,
                 booleanROImask,
                 booleanLabels):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_path = data_path
        self.n_classes = n_classes

        self.booleanFLAIR = booleanFLAIR
        self.booleanT1 = booleanT1
        self.booleanT1c = booleanT1c
        self.booleanT2 = booleanT2
        self.booleanROImask = booleanROImask
        self.booleanLabels = booleanLabels
        self.num_input_modalities = int(self.booleanFLAIR) + int(self.booleanT1) + int(self.booleanT1c) + \
                                    int(self.booleanT2)


        self.subject_list = self.create_subjects()

    def create_subjects(self):
        booleanFLAIR = self.booleanFLAIR
        booleanT1 = self.booleanT1
        booleanT1c = self.booleanT1c
        booleanT2 = self.booleanT2
        booleanROImask = self.booleanROImask
        booleanLabels = self.booleanLabels

        subject_list = []
        data_path = self.data_path
        if not isinstance(self.data_path, list):
            data_path = [data_path]

        for path in data_path:
            for subject in os.listdir(path):
                subject_list.append(Subject(path,
                                            subject,
                                            booleanFLAIR=booleanFLAIR,
                                            booleanT1=booleanT1,
                                            booleanT1c=booleanT1c,
                                            booleanT2=booleanT2,
                                            booleanROImask=booleanROImask,
                                            booleanLabels=booleanLabels
                                            ))

        return subject_list

    def get_subject(self,id):
        return filter(lambda subject: subject.id == id, self.subject_list)[0]

    def get_affine(self):
        subj = self.subject_list[0]
        return subj.get_affine()


    def set_subject_list(self, subject_list):
        self.subject_list = subject_list


class Dataset_train(Dataset):

    def __init__(self,
                 input_shape,
                 output_shape,
                 data_path,
                 n_classes,
                 n_subepochs,

                 sampling_scheme,
                 sampling_weights,

                 n_subjects_per_epoch_train,
                 n_segments_per_epoch_train,
                 n_subjects_per_epoch_validation,
                 n_segments_per_epoch_validation,

                 id_list_train,
                 id_list_validation,

                 booleanFLAIR,
                 booleanT1,
                 booleanT1c,
                 booleanT2,
                 booleanROImask,
                 booleanLabels,

                 data_augmentation_flag =False,
                 class_weights = 'constant'):

        self.n_subepochs = n_subepochs
        self._class_weights = class_weights
        self.sampling_scheme = sampling_scheme_selection(sampling_scheme, n_classes)
        self.sampling_scheme.set_probabilities(sampling_weights)

        self.n_segments_per_epoch_train = n_segments_per_epoch_train
        self.n_subjects_per_epoch_train = n_subjects_per_epoch_train
        self.n_segments_per_epoch_validation = n_segments_per_epoch_validation
        self.n_subjects_per_epoch_validation = n_subjects_per_epoch_validation

        self.id_list_train = id_list_train
        self.id_list_validation = id_list_validation
        self.subject_list = self.create_subjects()

        self.data_augmentation_flag = data_augmentation_flag

        super(Dataset_train,self).__init__(input_shape,
                                           output_shape,
                                           data_path,
                                           n_classes,
                                           booleanFLAIR,
                                           booleanT1,
                                           booleanT1c,
                                           booleanT2,
                                           booleanROImask,
                                           booleanLabels,
                                           )

    @property
    def class_weights(self):
        if self._class_weights == 'constant' or self._class_weights is None:
            return None
        elif self._class_weights == 'inverse_weights':
            return self.get_class_weights()
        else:
            raise ValueError("Please, specify a correct weighting of the cost function")

    def create_subjects(self):

        super(Dataset_train,self).create_subjects()
        self.subject_list = self._split_train_val(subject_list=subject_list)

        return self.subject_list

    def set_subject_list(self, subject_list):
        self.subject_list = self._split_train_val(subject_list=subject_list)

    def _split_train_val(self,subject_list):

        L = len(subject_list)
        np.random.seed(42)
        np.random.shuffle(subject_list)
        for subject in subject_list[:int(np.ceil(self.id_list_train*L))]:
                subject.train0_val1_both2 = 0
        for subject in subject_list[int(L - np.floor(self.id_list_validation * L)):]:
            subject.train0_val1_both2 = 1


        return subject_list
        

    def _get_subjects_subepoch(self, subject_list,n_subjects):
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
        index_fill_categories = np.random.choice(n_categories, size=n_segments % n_categories, replace=True,p=prob_sampling_samples)
        for index_cat in index_fill_categories:
            n_segments_per_category[index_cat] += 1

        # If spliting exact number of segments between classes, take one more of the positive class
        n_segments_distribution = np.zeros((n_subjects, n_categories),dtype = "int32")
        for index_cat in range(n_categories):
            n_segments_distribution[:,index_cat] = int(np.floor( n_segments_per_category[index_cat] / n_subjects ))
            np.random.seed(42)
            n_segments_distribution[np.random.randint(0, n_subjects, size=n_segments_per_category[index_cat] % n_subjects, dtype="int32"),
                                    index_cat] += 1

        return n_segments_distribution


    def load_image_segments(self,
                            subject_list,
                            n_subjects,
                            n_segments,
                            samplingInstance
                            ):

        subject_list = self._get_subjects_subepoch(subject_list, n_subjects)
        prob_sampling_samples = samplingInstance.get_probabilities()
        n_segments_distribution = self.sampling_parameters(n_subjects,
                                                           n_segments,
                                                           prob_sampling_samples
                                                           )


        image_segments = np.zeros((np.sum(n_segments_distribution), self.num_input_modalities,) + tuple(self.input_shape),dtype=np.float32)
        labels_patch_segments = np.zeros((np.sum(n_segments_distribution),) + tuple(self.output_shape),dtype=np.int32)

        # Centered in the input image if input_shape - output_shape is even. Otherwise it will be one voxel to the right
        gt_patch_width = (np.asarray(self.input_shape, dtype=int) - np.asarray(self.output_shape, dtype=int)) / 2

        cur_segment = 0
        for index, subject in enumerate(subject_list):
            image_channels = subject.load_channels() #4D array [channels, dim0,dim1,dim2]
            image_labels = subject.load_labels() #3D array [dim0,dim1,dim2]
            roi_mask = subject.load_ROI_mask()

            image_channels, image_labels, roi_mask = samplingInstance.get_proper_images(self.input_shape,
                                                                                        image_channels,
                                                                                        image_labels,
                                                                                        roi_mask)

            mask = samplingInstance.get_weighted_mask(self.input_shape, image_labels.shape,
                                                      ROI_mask=roi_mask,labels_mask=image_labels)

            central_voxels_coordinates, segments_coordinates = self._get_segments_coordinates(self.input_shape,
                                                                                              mask,
                                                                                              n_segments_distribution[index,:]
                                                                                              )

            if DEBUG:
                print 'Subject: ' + subject.id
                print 'Class distribution of subject: ' + str(np.bincount(image_labels.reshape(-1, ).astype(int), minlength=self.n_classes))


            for index_segment in range(central_voxels_coordinates.shape[1]):
                coor = segments_coordinates[:, index_segment, :]
                print 'b'
                print coor
                image_segments[cur_segment] = image_channels[:, coor[0, 0]:coor[0, 1] + 1, coor[1, 0]:coor[1, 1] + 1,
                                              coor[2, 0]:coor[2, 1] + 1]
                labels_whole_segment = image_labels[coor[0, 0]:coor[0, 1] + 1, coor[1, 0]:coor[1, 1] + 1,
                                       coor[2, 0]:coor[2, 1] + 1]

                labels_patch_segments[cur_segment] = labels_whole_segment[
                                                     gt_patch_width[0]:self.output_shape[0] + gt_patch_width[0],
                                                     gt_patch_width[1]:self.output_shape[1] + gt_patch_width[1],
                                                     gt_patch_width[2]:self.output_shape[2] + gt_patch_width[2]]
                cur_segment += 1


        image_segments,labels_patch_segments = self.data_augmentation(image_segments,labels_patch_segments,self.data_augmentation_flag)
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

            if DEBUG: print "Central voxels chosen in class: " + str(index_cat) + str(indices_central_voxels_flattened)

            coord_central_voxels = np.concatenate((coord_central_voxels,
                                                   np.asarray(np.unravel_index(indices_central_voxels_flattened,
                                                                        mask.shape))), axis=1)

        coord_segments = np.zeros(list(coord_central_voxels.shape) + [2], dtype='int32')
        coord_segments[:, :, 0] = coord_central_voxels - half_segment_dimensions[:, np.newaxis, 0]
        coord_segments[:, :, 1] = coord_central_voxels + half_segment_dimensions[:, np.newaxis, 1]

        return (coord_central_voxels, coord_segments)


    def data_generator(self, batch_size, train_val,subject_list = None):
        subj_list = self.subject_list if subject_list is None else subject_list

        while True:
            if train_val == 'train':
                subj_list_train = filter(lambda subject: subject.train0_val1_both2 == 0 or subject.train0_val1_both2 == 2,subj_list)
                n_segments_subepoch = self.n_segments_per_epoch_train / self.n_subepochs
                n_subjects = self.n_subjects_per_epoch_train

                for subepoch in range(self.n_subepochs):
                    np.random.shuffle(subj_list_train)
                    image_segments, label_segments = self.load_image_segments(subj_list_train,
                                                                                  n_subjects,
                                                                                  n_segments_subepoch,
                                                                                  self.sampling_scheme
                                                                                  )

                    for i in range(int(np.ceil(len(image_segments) / float(batch_size)))):
                        try:
                            features = image_segments[i * batch_size:(i + 1) * batch_size, :, :, :]
                            labels = one_hot_representation(label_segments[i * batch_size:(i + 1) * batch_size, :, :, :],
                                                            self.n_classes)
                        except:
                            features = image_segments[i * batch_size:, :, :, :]
                            labels = one_hot_representation(label_segments[i * batch_size:, :, :, :], self.n_classes)

                        yield features, labels


            elif train_val == 'val':

                subj_list_val = filter(lambda subject: subject.train0_val1_both2 == 1 or subject.train0_val1_both2 == 2,subj_list)
                np.random.shuffle(subj_list_val)
                n_segments = self.n_segments_per_epoch_validation
                n_subjects = self.n_subjects_per_epoch_validation

                image_segments, label_segments = self.load_image_segments(subj_list_val,
                                                                          n_subjects,
                                                                          n_segments,
                                                                          self.sampling_scheme)

                for num_batch in range(int(np.ceil(len(image_segments) / float(batch_size)))):
                    if (num_batch + 1) * batch_size > n_segments:
                        features = image_segments[num_batch * batch_size:, :, :, :]
                        labels = one_hot_representation(label_segments[num_batch * batch_size:, :, :, :], self.n_classes)

                    else:
                        features = image_segments[num_batch * batch_size:(num_batch + 1) * batch_size, :, :, :]
                        labels = one_hot_representation(
                            label_segments[num_batch * batch_size:(num_batch + 1) * batch_size, :, :, :],
                            self.n_classes)

                    yield features, labels

            else:
                raise ValueError("You should properly specify if train or validation")

    def data_generator_inference(self, train_val, subject_list=None, num_subjects=None):

        subj_list = self.subject_list if subject_list is None else subject_list
        if train_val == 'train':
            subj_list = filter(lambda subject: subject.train0_val1_both2 == 0 or subject.train0_val1_both2 == 2,
                               subj_list)
        elif train_val == 'val':
            subj_list = filter(lambda subject: subject.train0_val1_both2 == 1 or subject.train0_val1_both2 == 2,
                               subj_list)

        num_subjects = len(subj_list) if num_subjects is None else num_subjects
        subj_list = subj_list[:num_subjects]

        for subject in subj_list:
            image_channels = padding3D(subject.load_channels(),'multiple',32)[np.newaxis, :]  # 4D array [channels, dim0,dim1,dim2]
            labels = padding3D(subject.load_labels(),'multiple',32)[np.newaxis,:]

            yield (image_channels, one_hot_representation(labels,self.n_classes), subject.id)

    def data_generator_one(self, subject):
        image_channels = padding3D(subject.load_channels(), 'multiple', 32)[np.newaxis,:]  # 4D array [channels, dim0,dim1,dim2]
        labels = padding3D(subject.load_labels(), 'multiple', 32)[np.newaxis, :]

        return (image_channels, one_hot_representation(labels, self.n_classes), subject.id)

    def get_class_distribution(self, train_val='train'):
        subj_list = self.subject_list
        if train_val == 'train':
            subj_list = filter(lambda subject: subject.train0_val1_both2 == 0 or subject.train0_val1_both2 == 2,
                               subj_list)
        elif train_val == 'val':
            subj_list = filter(lambda subject: subject.train0_val1_both2 == 1 or subject.train0_val1_both2 == 2,
                               subj_list)
        else:
            raise ValueError('Please, specify wether you want the class_weight for the training or validation split')

        class_frequencies = np.zeros(self.n_classes)

        for subj in subj_list:
            labels = subj.load_labels()
            mask = subj.load_ROI_mask()
            class_frequencies += np.bincount(labels.flatten().astype('int'), weights=mask.flatten(),
                                             minlength=self.n_classes)

        return class_frequencies

    def get_class_weights(self,trainOrVal = 'train'):
        subj_list = self.subject_list
        if trainOrVal == 'train':
            subj_list = filter(lambda subject: subject.train0_val1_both2 == 0 or subject.train0_val1_both2 == 2,
                               subj_list)
        elif trainOrVal == 'val':
            subj_list = filter(lambda subject: subject.train0_val1_both2 == 1 or subject.train0_val1_both2 == 2,
                               subj_list)
        else:
            raise ValueError('Please, specify wether you want the class_weight for the training or validation split')

        class_frequencies = np.zeros(self.n_classes)

        for subj in subj_list:
            labels = subj.load_labels()
            mask = subj.load_ROI_mask()
            print 'a'
            print np.unique(labels)
            class_frequencies += np.bincount(labels.flatten().astype('int'), weights=mask.flatten(),
                                             minlength=self.n_classes)
        class_frequencies = class_frequencies / np.sum(class_frequencies)
        class_weight = np.sort(class_frequencies)[3] / class_frequencies

        return class_weight


    @staticmethod
    def data_augmentation(image,labels,data_augmentation_flag):

        if data_augmentation_flag:

            for index_segment in range(image.shape[0]):
                reflection = np.random.randint(0,2,size=3)*2-1
                image[index_segment] = image[index_segment,:,::reflection[0],::reflection[1],::reflection[2]]
                labels[index_segment] = labels[index_segment, ::reflection[0], ::reflection[1], ::reflection[2]]
            return image,labels
        else:
            return image,labels

    @staticmethod
    def data_shuffle(image,labels):
        index_shuffle = np.arange(image.shape[0])
        np.random.shuffle(index_shuffle)

        return (image[index_shuffle], labels[index_shuffle])



