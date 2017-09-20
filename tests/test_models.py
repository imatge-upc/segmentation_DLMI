import unittest
from src.layers import BatchNormalizationMasked
from database.BraTS.data_loader import Loader
from src.config import DB
from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization
import numpy as np


class TestLayers(unittest.TestCase):

    def setUp(self):
        brats_db = Loader.create(config_dict=DB.BRATS2017)
        subject_list = brats_db.load_subjects()
        self.subject = subject_list[0]


    def test_BNmasked(self):
        image = self.subject.load_channels()
        image = image[np.newaxis,:] + 1
        mask = self.subject.load_ROI_mask()
        mask = mask[np.newaxis,:,:,:,np.newaxis]
        mask = np.concatenate((mask,mask,mask,mask),axis=4)
        print(mask.shape)

        mean = np.sum(image*mask,axis=(0,1,2,3))/np.sum(mask,axis=(0,1,2,3))
        std = np.sqrt(np.sum(np.square((image - mean)*mask),axis=(0,1,2,3))/np.sum(mask,axis=(0,1,2,3)))

        image_normalized = ((image -mean) / std)*mask

        input_shape = self.subject.get_subject_shape() + (4,)

        x = Input(shape=input_shape, name='V-net_input')
        mask_brain = Input(shape=input_shape, name='V-net_mask1')

        y = BatchNormalizationMasked(axis=4,momentum=0.0,center=False, scale=False)([x,mask_brain])

        model = Model(inputs=[x,mask_brain],outputs=[y])
        model.compile('sgd', 'mse')

        # Predict data from the model
        for i in range(2):
            print(i)
            model.train_on_batch([image,mask],image_normalized)
        prediction = model.predict([image,mask], batch_size=1)

        print(model.get_weights())

        mean_pred = np.sum(prediction*mask,axis=(0,1,2,3))/np.sum(mask,axis=(0,1,2,3))
        std_pred = np.sqrt(np.sum(np.square((prediction - mean_pred)*mask),axis=(0,1,2,3))/np.sum(mask,axis=(0,1,2,3)))

        mean_true = np.sum(image_normalized * mask, axis=(0, 1, 2, 3)) / np.sum(mask, axis=(0, 1, 2, 3))
        std_true = np.sqrt(np.sum(np.square((image_normalized - mean_true)*mask), axis=(0, 1, 2, 3)) / np.sum(mask, axis=(0, 1, 2, 3)))

        print(mean)
        print(mean_true)
        print(mean_pred)
        print(np.mean(prediction))

        print(std)
        print(std_true)
        print(std_pred)


        # Assertions
        for i in range(4):
            self.assertAlmostEqual(mean_pred[i], 0.0, places = 3, msg='Predicted mean is different from 0.000XX')
            self.assertAlmostEqual(std_pred[i], 1.0, places = 2, msg='Predicted std is different from 1.000XX')





if __name__ == '__main__':
    unittest.main()
