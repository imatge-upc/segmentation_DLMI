import unittest
from src.losses import generalised_wasserstein_dice_loss as gwd
from src.activations import elementwise_softmax_3d
from keras.models import Sequential, Model
from keras.layers import Lambda, Activation
import numpy as np


class TestLayers(unittest.TestCase):

    def setUp(self):
        # Create fake data with known softmax distribution
        self.data_shape = (1, 2, 1, 1, 4)
        self.mock_x = np.zeros(self.data_shape)  # Input data
        mock_x_softmax = np.zeros(self.data_shape)  # Softmax from input data
        self.mock_y = np.zeros(self.data_shape)  # Labels
        distribution_1 = np.asarray([-1, -10, -1, 4])
        distribution_2 = np.asarray([-5, -20, 30, -1])
        self.mock_x[0, 0, 0, 0, :] = distribution_1
        self.mock_x[0, 1, 0, 0, :] = distribution_2
        mock_x_softmax[0, 0, 0, 0, :] = np.exp(distribution_1) / np.sum(np.exp(distribution_1))
        mock_x_softmax[0, 1, 0, 0, :] = np.exp(distribution_2) / np.sum(np.exp(distribution_2))
        self.mock_x_softmax = mock_x_softmax
        self.mock_y[0, 0, 0, 0, :] = np.asarray([0, 0, 0, 1])
        self.mock_y[0, 1, 0, 0, :] = np.asarray([0, 0, 1, 0])

    def test_GWD(self):
        # Compute categorical crossentropy
        indices = self.mock_y > 0
        selected_log = -np.log(self.mock_x_softmax[indices])
        self.loss = 0#np.sum(selected_log) / np.sum(self.mock_y)
        # Create keras model with this activation and compile it
        model = Sequential()
        activation_layer = Lambda(lambda x: x,
                                  input_shape=self.data_shape[1:],
                                  output_shape=self.data_shape[1:]
                                  )
        model.add(activation_layer)
        model.compile('sgd', loss=gwd)

        # Predict data from the model
        loss = model.evaluate(self.mock_y, self.mock_y, batch_size=1, verbose=0)
        # Assertions
        print('Expected loss: {}'.format(self.loss))
        print('Actual loss: {}'.format(loss))
        self.assertTrue(np.allclose(loss, self.loss),
                        msg='Categorical cross-entropy loss 3D does not produce the expected results')


if __name__ == '__main__':
    unittest.main()
