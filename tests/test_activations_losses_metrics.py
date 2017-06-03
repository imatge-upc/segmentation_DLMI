from __future__ import print_function
from _functools import partial
import unittest

import numpy as np
from keras.layers import Lambda, Dense, Convolution2D
from keras.models import Sequential
import keras.backend as K
from src.callbacks import GetActivationsLayer, GetWeightsLayer
from src.activations import elementwise_softmax_3d
from src.losses import categorical_crossentropy_3d, categorical_crossentropy_3d_mask
from src.metrics import dice, dice_whole, dice_core, dice_enhance, recall_0, accuracy_mask
from src.helpers.preprocessing_utils import one_hot_representation
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data


class TestElementwiseSoftmax3D(unittest.TestCase):
    def setUp(self):
        # Create fake data with known softmax distribution
        self.data_shape = (1, 5, 2, 1, 1)
        self.mock_data = np.zeros(self.data_shape)
        self.expected_result = np.zeros(self.data_shape)
        distribution_1 = np.asarray([-1, 0, -10, -1, 4])
        distribution_2 = np.asarray([-5, -10, -20, 30, -1])
        self.mock_data[0, :, 0, 0, 0] = distribution_1
        self.expected_result[0, :, 0, 0, 0] = np.exp(distribution_1) / np.sum(np.exp(distribution_1))
        self.mock_data[0, :, 1, 0, 0] = distribution_2
        self.expected_result[0, :, 1, 0, 0] = np.exp(distribution_2) / np.sum(np.exp(distribution_2))

    def test_elementwise_softmax_3d(self):
        # Create keras model with this activation and compile it
        model = Sequential()
        activation_layer = Lambda(elementwise_softmax_3d,
                                  input_shape=self.data_shape[1:],
                                  output_shape=self.data_shape[1:]
                                  )
        model.add(activation_layer)
        model.compile('sgd', 'mse')

        # Predict data from the model
        prediction = model.predict(self.mock_data, batch_size=1)

        # Assertions
        self.assertTupleEqual(prediction.shape, self.expected_result.shape,
                              msg='Output from activation has expected shape')

        self.assertTrue(np.allclose(prediction, self.expected_result),
                        msg='Softmax activation does not produce the expected results')


class TestCategoricalCrossentropy(unittest.TestCase):
    def setUp(self):
        # Create fake data with known softmax distribution
        self.data_shape = (1, 5, 2, 1, 1)
        self.mock_x = np.zeros(self.data_shape)  # Input data
        mock_x_softmax = np.zeros(self.data_shape)  # Softmax from input data
        self.mock_y = np.zeros(self.data_shape)  # Labels
        distribution_1 = np.asarray([-1, 0, -10, -1, 4])
        distribution_2 = np.asarray([-5, -10, -20, 30, -1])
        self.mock_x[0, :, 0, 0, 0] = distribution_1
        self.mock_x[0, :, 1, 0, 0] = distribution_2
        mock_x_softmax[0, :, 0, 0, 0] = np.exp(distribution_1) / np.sum(np.exp(distribution_1))
        mock_x_softmax[0, :, 1, 0, 0] = np.exp(distribution_2) / np.sum(np.exp(distribution_2))
        self.mock_x_softmax = mock_x_softmax
        self.mock_y[0, :, 0, 0, 0] = np.asarray([0, 0, 0, 0, 1])
        self.mock_y[0, :, 1, 0, 0] = np.asarray([0, 0, 0, 1, 0])


    def test_categorical_crossentropy(self):
        # Compute categorical crossentropy
        indices = self.mock_y > 0
        selected_log = -np.log( self.mock_x_softmax[indices])
        self.loss = np.sum(selected_log) / np.sum(self.mock_y)
        # Create keras model with this activation and compile it
        model = Sequential()
        activation_layer = Lambda(elementwise_softmax_3d,
                                  input_shape=self.data_shape[1:],
                                  output_shape=self.data_shape[1:]
                                  )
        model.add(activation_layer)
        model.compile('sgd', categorical_crossentropy_3d())

        # Predict data from the model
        loss = model.evaluate(self.mock_x, self.mock_y, batch_size=1, verbose=0)
        # Assertions
        print('Expected loss: {}'.format(self.loss))
        print('Actual loss: {}'.format(loss))
        self.assertTrue(np.allclose(loss, self.loss),
                        msg='Categorical cross-entropy loss 3D does not produce the expected results')

    def test_categorical_crossentropy_weighted(self):
        # Compute categorical crossentropy
        w = np.asarray([0.25, 0.25, 0.25, 0.125, 0.125])
        sample_weight = w[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        indices = self.mock_y > 0

        log = -np.log(self.mock_x_softmax)*sample_weight
        selected_log = log[indices]
        self.loss = np.sum(selected_log) / np.sum(self.mock_y)

        # Create keras model with this activation and compile it

        model = Sequential()
        activation_layer = Lambda(elementwise_softmax_3d,
                                  input_shape=self.data_shape[1:],
                                  output_shape=self.data_shape[1:]
                                  )
        model.add(activation_layer)
        model.compile('sgd', loss=categorical_crossentropy_3d(sample_weights=K.variable(w)))

        # Predict data from the model
        loss = model.evaluate(self.mock_x, self.mock_y, batch_size=1, verbose=0)
        # Assertions
        print('Expected loss: {}'.format(self.loss))
        print('Actual loss: {}'.format(loss))
        self.assertTrue(np.allclose(loss, self.loss),
                        msg='Categorical cross-entropy loss 3D does not produce the expected results')



class TestDiceScore(unittest.TestCase):
    def setUp(self):
        # Create fake data with known softmax distribution
        self.data_shape = (1, 5, 2, 1, 1)
        self.mock_x = np.zeros(self.data_shape)  # Input data
        mock_x_softmax = np.zeros(self.data_shape)  # Softmax from input data
        self.mock_y = np.zeros(self.data_shape)  # Labels
        distribution_1 = np.asarray([-1, 0, -10, -1, 4])
        distribution_2 = np.asarray([-5, -10, -20, 30, -1])
        self.mock_x[0, :, 0, 0, 0] = distribution_1
        self.mock_x[0, :, 1, 0, 0] = distribution_2
        mock_x_softmax[0, :, 0, 0, 0] = np.exp(distribution_1) / np.sum(np.exp(distribution_1))
        mock_x_softmax[0, :, 1, 0, 0] = np.exp(distribution_2) / np.sum(np.exp(distribution_2))
        self.mock_y[0, :, 0, 0, 0] = np.asarray([0, 0, 0, 0, 1])
        self.mock_y[0, :, 1, 0, 0] = np.asarray([0, 0, 0, 1, 0])
        # Compute dice score
        binarized_prediction = mock_x_softmax / np.max(mock_x_softmax, axis=1, keepdims=True)
        binarized_prediction = np.asarray(binarized_prediction, dtype=np.int8)
        intersection = np.sum(binarized_prediction * self.mock_y)
        self.dice_score = (2 * intersection) / (np.sum(self.mock_y) + np.sum(binarized_prediction))



    def test_dice_whole_score(self):
        # Create keras model with this activation and compile it
        y_true = np.reshape(np.asarray([0, 0, 2, 1, 0, 1, 4, 0, 0, 3, 3, 0, 4, 1, 0, 0]),(2,2,2,2))
        y_true_onehot = K.variable(one_hot_representation(y_true,5))
        y_pred = np.reshape(np.asarray([0, 1, 2, 0, 0, 1, 4, 0, 3, 3, 1, 0, 4, 1, 2, 0]),(2,2,2,2))
        y_pred_onehot = K.variable(one_hot_representation(y_pred,5)*0.6)
        dice_score = 2.0 * 7 / (8 + 10)

        # Assertions
        print('Expected Dice: {}'.format(K.eval(dice_whole(y_true_onehot, y_pred_onehot))))
        print('Actual Dice: {}'.format(dice_score))
        self.assertTrue(np.allclose(dice_score, K.eval(dice_whole(y_true_onehot, y_pred_onehot)), rtol=1.e-5, atol=1.e-5),
                        msg='Dice whole 3D does not produce the expected results')


    def test_dice_core_score(self):
        # Create keras model with this activation and compile it
        y_true = np.reshape(np.asarray([0, 0, 2, 1, 0, 1, 4, 0, 0, 3, 3, 0, 4, 1, 0, 0]),(2,2,2,2))
        y_true_onehot = K.variable(one_hot_representation(y_true,5))
        y_pred = np.reshape(np.asarray([0, 1, 2, 0, 0, 1, 4, 0, 3, 3, 1, 0, 4, 1, 2, 0]),(2,2,2,2))
        y_pred_onehot = K.variable(one_hot_representation(y_pred,5)*0.6)
        dice_score = 2.0 * 6 / (8 + 7)

        # Assertions
        print('Expected Dice: {}'.format(K.eval(dice_core(y_true_onehot, y_pred_onehot))))
        print('Actual Dice: {}'.format(dice_score))
        self.assertTrue(np.allclose(dice_score, K.eval(dice_core(y_true_onehot, y_pred_onehot)), rtol=1.e-5, atol=1.e-5),
                        msg='Dice score within the core lables in 3D does not produce the expected results')


    def test_dice_enchance_score(self):
        # Create keras model with this activation and compile it
        y_true = np.reshape(np.asarray([0, 0, 2, 1, 0, 1, 4, 0, 0, 3, 3, 0, 4, 1, 0, 0]), (2, 2, 2, 2))
        y_true_onehot = K.variable(one_hot_representation(y_true, 5))
        y_pred = np.reshape(np.asarray([0, 1, 2, 0, 0, 1, 4, 0, 3, 3, 1, 0, 4, 1, 2, 0]), (2, 2, 2, 2))
        y_pred_onehot = K.variable(one_hot_representation(y_pred, 5)*0.6)
        dice_score = 2.0 * 1 / (2 + 2)

        # Assertions
        # Assertions
        print('Expected Dice: {}'.format(K.eval(dice_enhance(y_true_onehot, y_pred_onehot))))
        print('Actual Dice: {}'.format(dice_score))
        self.assertTrue(np.allclose(dice_score, K.eval(dice_enhance(y_true_onehot, y_pred_onehot)), rtol=1.e-5, atol=1.e-5),
                        msg='Dice score within the enhancing class in 3D does not produce the expected results')


    def test_recall_0(self):
        # Create keras model with this activation and compile it
        y_true = np.reshape(np.asarray([0, 0, 2, 1, 0, 1, 4, 0, 0, 3, 3, 0, 4, 1, 0, 0]), (2, 2, 2, 2))
        y_true_onehot = K.variable(one_hot_representation(y_true, 5))
        y_pred = np.reshape(np.asarray([0, 1, 2, 0, 0, 1, 4, 0, 3, 3, 1, 0, 4, 1, 2, 0]), (2, 2, 2, 2))
        y_pred_onehot = K.variable(one_hot_representation(y_pred, 5)*0.6)
        recall0_score = 5.0 / 8.0

        # Assertions
        print('Expected Dice: {}'.format(K.eval(recall_0(y_true_onehot, y_pred_onehot))))
        print('Actual Dice: {}'.format(recall0_score))
        self.assertTrue(np.allclose(recall0_score, K.eval(recall_0(y_true_onehot, y_pred_onehot)), rtol=1.e-5, atol=1.e-5),
                        msg='Recall with non-tumor are as positive in 3D does not produce the expected results')


    def test_accuracy_mask(self):
        # Create keras model with this activation and compile it
        y_true = np.reshape(np.asarray([0, 0, 2, 1, 0, 1, 4, 0, 0, 3, 3, 0, 4, 1, 0, 0]), (1, 4, 2, 2))
        y_true_onehot = K.variable(one_hot_representation(y_true, 5))
        y_pred = np.reshape(np.asarray([0, 1, 2, 0, 0, 1, 4, 0, 3, 3, 1, 0, 4, 1, 2, 0]), (1, 4, 2, 2))
        y_pred_onehot = K.variable(one_hot_representation(y_pred, 5) * 0.6)
        mask = np.zeros((1,4,2,2), dtype='int32')
        mask[:,:2,:,:] = 1
        accuracy_score = 6.0 / 8.0
        accuracy = accuracy_mask(mask)

        # Assertions
        print('Expected Dice: {}'.format(K.eval(accuracy(y_true_onehot, y_pred_onehot))))
        print('Actual Dice: {}'.format(accuracy_score))
        self.assertTrue(
            np.allclose(accuracy_score, K.eval(accuracy(y_true_onehot, y_pred_onehot)), rtol=1.e-5, atol=1.e-5),
            msg='Dice whole 3D does not produce the expected results')

class TestCallbacks(unittest.TestCase):

    def setUp(self):

        self.input_dim = 5
        self.nb_hidden = 4
        self.nb_class = 2
        self.batch_size = 5
        self.train_samples = 20
        self.test_samples = 20

    # def test_callback_save_activations(self):
    #     (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=self.train_samples,
    #                                                          nb_test=self.test_samples,
    #                                                          input_shape=(self.input_dim,),
    #                                                          classification=True,
    #                                                          nb_class=self.nb_class)
    #
    #     y_test = np_utils.to_categorical(y_test)
    #     y_train = np_utils.to_categorical(y_train)
    #     # case 1
    #     monitor = 'val_loss'
    #     save_best_only = False
    #     mode = 'auto'
    #
    #     model = Sequential()
    #     model.add(Dense(self.nb_hidden, input_dim=self.input_dim, activation='relu'))
    #     model.add(Dense(self.nb_hidden, activation='relu'))
    #     model.add(Dense(self.nb_hidden, activation='relu'))
    #     model.add(Dense(self.nb_class, activation='softmax'))
    #     model.compile(loss='mse',
    #                   optimizer='sgd',
    #                   metrics=['accuracy'])
    #
    #     cbks = [GetActivationsLayer(layer_num=0)]
    #     model.fit(X_train, y_train, batch_size=self.batch_size,
    #               validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)

    def test_callback_save_weights(self):
            (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=self.train_samples,
                                                                 nb_test=self.test_samples,
                                                                 input_shape=(self.input_dim,),
                                                                 classification=True,
                                                                 nb_class=self.nb_class)

            print(X_train.shape)
            y_test = np_utils.to_categorical(y_test)
            y_train = np_utils.to_categorical(y_train)
            # case 1
            monitor = 'val_loss'
            save_best_only = False
            mode = 'auto'

            model = Sequential()
            model.add(Convolution2D(self.nb_hidden, 3,3,  activation='relu', input_shape = X_train.shape))
            model.add(Convolution2D(self.nb_hidden, 3, 3, activation='relu'))
            model.add(Dense(self.nb_class, activation='softmax'))
            model.compile(loss='mse',
                          optimizer='sgd',
                          metrics=['accuracy'])

            cbks = [GetWeightsLayer(layer_num=1)]
            model.fit(X_train, y_train, batch_size=self.batch_size,
                      validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)


if __name__ == '__main__':
    unittest.main()
