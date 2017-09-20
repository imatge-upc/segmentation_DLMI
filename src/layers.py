from keras import backend as K
import tensorflow as tf
from keras.layers import Layer, BatchNormalization, InputSpec
from tensorflow.python.training import moving_averages



def repeat_channels(rep):
    def funct(x):
        return K.repeat_elements(x, rep, axis=4)
    return funct

def repeat_channels_shape(rep):
    def funct(input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]*rep)
    return funct

def repeat_slices(rep):
    def funct(x):
        return K.repeat_elements(x, rep, axis=3)
    return funct

def repeat_slices_shape(rep):
    def funct(input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3]*rep, input_shape[4])
    return funct

def complementary_mask(x):
    return K.ones_like(x) - x


def mask_tensor(x):
    tensor, mask = x
    rep = K.int_shape(tensor)[4]

    return tensor*K.repeat_elements(mask, rep, axis=4)


def fill_background_mask(x):
    tensor, mask = x
    rep = K.int_shape(tensor)[4] - 1
    full_mask = K.ones_like(mask) - mask
    K.repeat_elements(mask, rep, axis=4)

    return tensor + full_mask


def dice_1(y):
    """
    Computes the Sorensen-Dice metric, where P come from class 1,2,3,4,5
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : keras.placeholder
        Placeholder that contains the ground truth labels of the classes
    y_pred : keras.placeholder
        Placeholder that contains the class prediction

    Returns
    -------
    scalar
        Dice metric
    """
    y_true, y_pred = y
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :,1]
    mask_pred = y_pred_decision[:, :, :, :, 1]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_2(y):
    """
    Computes the Sorensen-Dice metric, where P come from class 1,2,3,4,5
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : keras.placeholder
        Placeholder that contains the ground truth labels of the classes
    y_pred : keras.placeholder
        Placeholder that contains the class prediction

    Returns
    -------
    scalar
        Dice metric
    """
    y_true, y_pred = y
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :, 2]
    mask_pred = y_pred_decision[:, :, :, :, 2]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())


def dice_3(y):
    """
    Computes the Sorensen-Dice metric, where P come from class 1,2,3,4,0
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : keras.placeholder
        Placeholder that contains the ground truth labels of the classes
    y_pred : keras.placeholder
        Placeholder that contains the class prediction

    Returns
    -------
    scalar
        Dice metric
    """
    y_true, y_pred = y
    y_pred_decision = tf.floor(y_pred / K.max(y_pred, axis=4, keepdims=True))

    mask_true = y_true[:, :, :, :, 3]
    mask_pred = y_pred_decision[:, :, :, :, 3]

    y_sum = K.sum(mask_true * mask_pred)

    return (2. * y_sum + K.epsilon()) / (K.sum(mask_true) + K.sum(mask_pred) + K.epsilon())



class BatchNormalizationMasked(BatchNormalization):
    def build(self, inputs_shape):
        input_shape = inputs_shape[0]
        super(BatchNormalizationMasked, self).build(input_shape)
    
    def call(self, inputs, training=None):
        input = inputs[0]
        super(BatchNormalizationMasked, self).call(input)

    # def build(self, inputs_shape):
    #     input_shape = inputs_shape[0]
    #     dim = input_shape[self.axis]
    #     if dim is None:
    #         raise ValueError('Axis ' + str(self.axis) + ' of '
    #                                                     'input tensor should have a defined dimension '
    #                                                     'but the layer received an input with shape ' +
    #                          str(input_shape) + '.')
    #     self.input_spec = [InputSpec(ndim=len(input_shape), axes={self.axis: dim}),
    #                        InputSpec(ndim=len(input_shape), axes={self.axis: dim})]
    #
    #     shape = (dim,)
    #
    #     if self.scale:
    #         self.gamma = self.add_weight(shape=shape,
    #                                      name='gamma',
    #                                      initializer=self.gamma_initializer,
    #                                      regularizer=self.gamma_regularizer,
    #                                      constraint=self.gamma_constraint)
    #     else:
    #         self.gamma = None
    #     if self.center:
    #         self.beta = self.add_weight(shape=shape,
    #                                     name='beta',
    #                                     initializer=self.beta_initializer,
    #                                     regularizer=self.beta_regularizer,
    #                                     constraint=self.beta_constraint)
    #     else:
    #         self.beta = None
    #     self.moving_mean = self.add_weight(
    #         shape=shape,
    #         name='moving_mean',
    #         initializer=self.moving_mean_initializer,
    #         trainable=False)
    #     self.moving_variance = self.add_weight(
    #         shape=shape,
    #         name='moving_variance',
    #         initializer=self.moving_variance_initializer,
    #         trainable=False)
    #     self.built = True
    #
    # def call(self, inputs, training=None):
    #     input, mask = inputs
    #     input_shape = K.int_shape(input)
    #
    #     # Prepare broadcasting shape.
    #     ndim = len(input_shape)
    #     reduction_axes = list(range(len(input_shape)))
    #     del reduction_axes[self.axis]
    #     broadcast_shape = [1] * len(input_shape)
    #     broadcast_shape[self.axis] = input_shape[self.axis]
    #
    #     # Determines whether broadcasting is needed.
    #     needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])
    #
    #     def normalize_inference():
    #         if needs_broadcasting:
    #
    #             # In this case we must explicitly broadcast all parameters.
    #             broadcast_moving_mean = K.reshape(self.moving_mean,
    #                                               broadcast_shape)
    #             broadcast_moving_variance = K.reshape(self.moving_variance,
    #                                                   broadcast_shape)
    #             if self.center:
    #                 broadcast_beta = K.reshape(self.beta, broadcast_shape)
    #             else:
    #                 broadcast_beta = None
    #             if self.scale:
    #                 broadcast_gamma = K.reshape(self.gamma,
    #                                             broadcast_shape)
    #             else:
    #                 broadcast_gamma = None
    #             return K.batch_normalization(
    #                 input,
    #                 broadcast_moving_mean,
    #                 broadcast_moving_variance,
    #                 broadcast_beta,
    #                 broadcast_gamma,
    #                 epsilon=self.epsilon)
    #         else:
    #             return K.batch_normalization(
    #                 input,
    #                 self.moving_mean,
    #                 self.moving_variance,
    #                 self.beta,
    #                 self.gamma,
    #                 epsilon=self.epsilon)
    #
    #     # If the learning phase is *static* and set to inference:
    #     if training in {0, False}:
    #         return normalize_inference()*mask
    #
    #     # If the learning is either dynamic, or set to training:
    #     normed_training, mean, variance = self._normalize_batch_in_training(
    #         input, mask, self.gamma, self.beta, reduction_axes,epsilon=self.epsilon)
    #
    #
    #     self.add_update([self._moving_average_update(self.moving_mean,
    #                                                  mean,
    #                                                  self.momentum,
    #                                                  ),#zero_debias = True),
    #                      self._moving_average_update(self.moving_variance,
    #                                                  variance,
    #                                                  self.momentum)],
    #                     inputs)
    #
    #     # Pick the normalized form corresponding to the training phase.
    #     return K.in_train_phase(normed_training,
    #                             normalize_inference,
    #                             training=training)
    #
    # @staticmethod
    # def _normalize_batch_in_training(x, mask, gamma, beta,
    #                                 reduction_axes, epsilon=1e-3):
    #     """Computes mean and std for batch then apply batch_normalization on batch.
    #     # Arguments
    #         x: Input tensor or variable.
    #         gamma: Tensor by which to scale the input.
    #         beta: Tensor with which to center the input.
    #         reduction_axes: iterable of integers,
    #             axes over which to normalize.
    #         epsilon: Fuzz factor.
    #     # Returns
    #         A tuple length of 3, `(normalized_tensor, mean, variance)`.
    #     """
    #     mean = 1.0*K.sum(x*mask, axis=reduction_axes)/K.sum(mask, axis=reduction_axes)
    #     var = 1.0/K.sum(mask, axis=reduction_axes)*K.sum(K.square((x-mean)*mask), axis=reduction_axes)
    #
    #     if sorted(reduction_axes) == list(range(K.ndim(x)))[:-1]:
    #         normed = tf.nn.batch_normalization(x, mean, var,
    #                                            beta, gamma,
    #                                            epsilon)
    #     else:
    #         # need broadcasting
    #         target_shape = []
    #         for axis in range(K.ndim(x)):
    #             if axis in reduction_axes:
    #                 target_shape.append(1)
    #             else:
    #                 target_shape.append(tf.shape(x)[axis])
    #         target_shape = tf.stack(target_shape)
    #
    #         broadcast_mean = tf.reshape(mean, target_shape)
    #         broadcast_var = tf.reshape(var, target_shape)
    #         if gamma is None:
    #             broadcast_gamma = None
    #         else:
    #             broadcast_gamma = tf.reshape(gamma, target_shape)
    #         if beta is None:
    #             broadcast_beta = None
    #         else:
    #             broadcast_beta = tf.reshape(beta, target_shape)
    #         normed = tf.nn.batch_normalization(x, broadcast_mean, broadcast_var,
    #                                            broadcast_beta, broadcast_gamma,
    #                                            epsilon)
    #
    #     normed = normed * mask
    #     return normed, mean, var
    #
    # @staticmethod
    # def _moving_average_update(x, value, momentum, zero_debias=False):
    #     """Compute the moving average of a variable.
    #
    #     # Arguments
    #         x: A Variable.
    #         value: A tensor with the same shape as `variable`.
    #         momentum: The moving average momentum.
    #
    #     # Returns
    #         An Operation to update the variable."""
    #     return moving_averages.assign_moving_average(
    #         x, value, momentum, zero_debias=zero_debias)

class Conditional(Layer):

    def call(self, inputs, training=None):
        return K.in_train_phase(inputs[0],inputs[1],training=training)
