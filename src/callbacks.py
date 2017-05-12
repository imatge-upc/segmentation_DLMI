import numpy as np
from keras import backend as K
from keras.callbacks import Callback


class PredefinedLearningRateDecay(Callback):
    def __init__(self, decay_rate, predefined_epochs=None):
        self.decay_rate = float(decay_rate)
        self.predefined_epochs = predefined_epochs if predefined_epochs is not None else [-1]

        super(PredefinedLearningRateDecay, self).__init__()

    def on_epoch_end(self, epoch, logs={}):

        if epoch in self.predefined_epochs or -1 in self.predefined_epochs:
            lr = self.model.optimizer.lr.get_value() / self.decay_rate
        else:
            lr = self.model.optimizer.lr.get_value()

        self.model.optimizer.lr.set_value(lr)


class LearningRateDecayAccuracyPlateaus(Callback):
    def __init__(self,decay_rate):
        self.decay_rate = float(decay_rate)

        super(LearningRateDecayAccuracyPlateaus, self).__init__()


    def on_train_begin(self, logs={}):
        self.wait = 0  # Allow instances to be re-used
        self.last = -np.Inf
        self.accumulation = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('loss')
        if current > self.last:
            self.accumulation += 1
            if self.accumulation > 1:
                lr = self.model.optimizer.lr.get_value() / self.decay_rate
            else:
                lr = self.model.optimizer.lr.get_value()
        else:
            lr =  self.model.optimizer.lr.get_value()
            self.accumulation = 0

        self.last = current
        self.model.optimizer.lr.set_value(lr)


