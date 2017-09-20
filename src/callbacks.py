import numpy as np
from keras import backend as K
from keras.callbacks import Callback
import math

class LearningRatePredefinedDecay(Callback):
    def __init__(self, decay_rate, predefined_epochs=None):
        self.decay_rate = float(decay_rate)
        self.predefined_epochs = predefined_epochs if predefined_epochs is not None else [-1]

        super(LearningRatePredefinedDecay, self).__init__()

    def on_epoch_end(self, epoch, logs={}):

        if epoch in self.predefined_epochs or -1 in self.predefined_epochs:
            lr = K.get_value(self.model.optimizer.lr) / self.decay_rate
            K.set_value(self.model.optimizer.lr, lr)



class LearningRateExponentialDecay(Callback):
    def __init__(self,epoch_n,num_epoch, ):
        self.epoch_n = epoch_n
        self.num_epoch = num_epoch
        super(LearningRateExponentialDecay, self).__init__()


    def on_train_begin(self, logs={}):
        self.lr_init = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.epoch_n:
            ratio = 1.0 * (self.num_epoch - epoch)  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (self.num_epoch - self.epoch_n))
            lr = np.float32(self.lr_init * ratio)
            K.set_value(self.model.optimizer.lr,lr)

class LearningRateDecay(Callback):
    def __init__(self, epoch_n=0, decay=0.1):
        self.epoch_n = epoch_n
        self.decay = decay
        super(LearningRateDecay, self).__init__()

    def on_train_begin(self, logs={}):
        self.lr_init = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.epoch_n:
            ratio = 1.0 - self.decay*(epoch - self.epoch_n )
            ratio = max(0, ratio)
            lr = np.float32(self.lr_init * ratio)
            K.set_value(self.model.optimizer.lr, lr)

