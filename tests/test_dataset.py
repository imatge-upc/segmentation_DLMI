import unittest
import numpy as np
from nose.plugins.attrib import attr
from src.dataset import Dataset_train
from src.helpers.Subject import Subject
from src.helpers.preprocessing_utils import one_hot_representation
from src.helpers.io_utils import *
from keras.utils.np_utils import to_categorical
from src.helpers.sampling import FBScheme, WholeScheme

class TestSubject(unittest.TestCase):
    def test_init(self):
        subj = Subject('/projects/neuro/BRATS/BRATS2015_Training/LGG',
                       'brats_2013_pat0001_1',
                       booleanFLAIR=False,
                       booleanT1=False,
                       booleanT2=False)

        self.assertIsNone(subj.T2_FILE)
        self.assertIsNone(subj.FLAIR_FILE)
        self.assertIsNone(subj.T1_FILE)
        self.assertEquals('/projects/neuro/BRATS/BRATS2015_Training/LGG/brats_2013_pat0001_1/'
                         'T1c.nii.gz',subj.T1c_FILE)
        self.assertEquals('/projects/neuro/BRATS/BRATS2015_Training/LGG/brats_2013_pat0001_1/'
                         'GT.nii.gz',subj.GT_FILE)


class TestLoadSubjects(unittest.TestCase):

    def setUp(self):
        self.subj = Subject('/mnt/imatge-projects/neuro/BRATS/BRATS2015_Training/LGG',
                       'brats_2013_pat0001_1',
                       booleanFLAIR=False,
                       booleanT1=False,
                       booleanT2=False)

    def test_load_channels(self):
        image_channels = self.subj.load_channels()
        self.assertEqual((1,240,240,155),image_channels.shape)

    def test_load_labels(self):
        labels = self.subj.load_labels()
        random_indices = np.random.rand(100,3)
        random_indices[:,0] = np.floor(random_indices[:,0] * 240)
        random_indices[:,1] = np.floor(random_indices[:,1] * 240)
        random_indices[:,2] = np.floor(random_indices[:,2] * 155)
        for i in range(random_indices.shape[0]):
            self.assertIn(labels[tuple(random_indices[i,:])],[0,1,2,3,4])



class TestBratsDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_fb=Dataset_train(input_shape=[25,25,25],
                                output_shape = [9,9,9],
                                data_path='/mnt/imatge-projects/neuro/BRATS/BRATS2015_Training/LGG',
                                n_classes=5,
                                n_subepochs=1,

                                sampling_scheme='foreground-background',
                                sampling_weights=[0.5,0.5],

                                n_subjects_per_epoch_train=2,
                                n_segments_per_epoch_train=40,
                                n_subjects_per_epoch_validation = 2,
                                n_segments_per_epoch_validation = 20,

                                id_list_train= 0.6,
                                id_list_validation = 0.4,
                                booleanFLAIR = True,
                                booleanT1 = True,
                                booleanT1c = True,
                                booleanT2 = True,
                                booleanROImask = True,
                                booleanLabels = True,

                                data_augmentation_flag=False,
                                class_weights='constant'

                                )
        self.dataset_whole =  Dataset_train(input_shape=[256, 256, 160],
                                      output_shape=[256, 256, 160],
                                      data_path='/mnt/imatge-projects/neuro/BRATS/BRATS2015_Training/LGG',
                                      n_classes=5,
                                      n_subepochs=1,

                                      sampling_scheme='whole',
                                      sampling_weights=[1.0],

                                      n_subjects_per_epoch_train=2,
                                      n_segments_per_epoch_train=2,
                                      n_subjects_per_epoch_validation=2,
                                      n_segments_per_epoch_validation=2,

                                      id_list_train=0.6,
                                      id_list_validation=0.4,
                                      booleanFLAIR=True,
                                      booleanT1=True,
                                      booleanT1c=True,
                                      booleanT2=True,
                                      booleanROImask=True,
                                      booleanLabels=True,

                                      data_augmentation_flag=False,
                                      class_weights='constant'

                                      )
        self.subj = Subject(data_path='/mnt/imatge-projects/neuro/BRATS/BRATS2015_Training/LGG',
                            subject_name='brats_2013_pat0001_1',
                            booleanFLAIR=True,
                            booleanT1=True,
                            booleanT2=True)

        self.subj2 = Subject('/mnt/imatge-projects/neuro/BRATS/BRATS2015_Training/LGG',
                            'brats_2013_pat0002_1',
                            booleanFLAIR=True,
                            booleanT1=True,
                            booleanT2=True)

        self.sampling_instance_fb = FBScheme(5)
        self.sampling_instance_fb.set_probabilities([0.5,0.5])
        self.sampling_instance_whole = WholeScheme(5)
        self.sampling_instance_whole.set_probabilities([1.0])


    def test_createSubjects(self):
        subj_list = self.dataset_fb.create_subjects()
        subj_list = filter(lambda subj: subj._id in ['LGG_brats_2013_pat0001_1', 'LGG_brats_2013_pat0002_1'], subj_list)

        self.assertEquals(2,len(subj_list))
        self.assertEquals('/mnt/imatge-projects/neuro/BRATS/BRATS2015_Training/LGG/brats_2013_pat0001_1/'
                          'T1c.nii.gz', subj_list[0].T1c_FILE)

        subj_list = self.dataset_fb.create_subjects()
        subj_list_train = filter(lambda subj: subj.train0_val1_both2 == 0 or subj.train0_val1_both2 == 2 ,subj_list)
        subj_list_val = filter(lambda subj: subj.train0_val1_both2 == 1 or subj.train0_val1_both2 == 2 ,subj_list)
        self.assertEquals(np.ceil(0.6*len(subj_list)),len(subj_list_train))
        self.assertEquals(np.floor(0.4*len(subj_list)), len(subj_list_val))

    def test_loadImageSegments(self):


        subject_list = self.dataset_fb.create_subjects()
        image_segments, labels_segments = self.dataset_fb.load_image_segments(subject_list,2,40,self.sampling_instance_fb)

        self.assertEquals((40,4,25,25,25),np.asarray(image_segments).shape)
        self.assertEquals((40,9,9,9),np.asarray(labels_segments).shape)
        self.assertEquals( 20,len( np.where(labels_segments[:,4,4,4] ==0)[0] )   )

    def test_loadGTLabels(self):
        subject_list = self.dataset_fb.create_subjects()
        image_segments, labels_segments = self.dataset_fb.load_image_segments(subject_list, 2, 40, self.sampling_instance_fb)

        one_hot1 = to_categorical(np.reshape(labels_segments,(-1,1)),5)
        one_hot2 = one_hot_representation(np.reshape(labels_segments,(-1,)),5)
        self.assertTrue(np.allclose(one_hot1,one_hot2))

    def test_get_weights(self):
        w = self.dataset_fb.get_class_weights('train')
        self.assertEquals(5,len(w))

    def test_dataGenerator(self):
        data_generator = self.dataset_fb.data_generator(batch_size=6,train_val='train')
        length = 0
        iteration = 0
        for data in data_generator:
            length += data[0].shape[0]
            if iteration == np.floor(40.0/6.0):
                break
            self.assertEquals((6,4,25,25,25),data[0].shape)
            self.assertEquals((6,5,9,9,9), data[1].shape)
            iteration += 1

        self.assertEquals(length, 40)

    def test_wholeSampling(self):
        subject_list = self.dataset_whole.create_subjects()
        image_segments, labels_segments = self.dataset_whole.load_image_segments(subject_list, 2, 2,
                                                                              self.sampling_instance_whole)

        self.assertEquals((2, 4, 256, 256, 160), np.asarray(image_segments).shape)
        self.assertEquals((2, 256, 256, 160), np.asarray(labels_segments).shape)
