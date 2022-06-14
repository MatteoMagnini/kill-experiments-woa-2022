from pathlib import Path
import numpy as np
from tensorflow.python.keras.callbacks import Callback

PATH = Path(__file__).parents[0]


class Conditions(Callback):
    def __init__(self, train_x, train_y, patience: int = 5, threshold: float = 0.25, stop_threshold_1: float = 0.99,
                 stop_threshold_2: float = 0.9):
        super(Conditions, self).__init__()
        self.train_x = train_x
        train_y = train_y.iloc[:, 0]
        self.train_y = np.zeros((train_y.size, train_y.max() + 1))
        self.train_y[np.arange(train_y.size), train_y] = 1
        self.patience = patience
        self.threshold = threshold
        self.stop_threshold_1 = stop_threshold_1
        self.stop_threshold_2 = stop_threshold_2
        self.best_acc = 0
        self.wait = 0
        self.best_weights = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        self.best_acc = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Second condition
        acc = logs.get('accuracy')
        if self.best_acc > acc > self.stop_threshold_2:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
        else:
            self.best_acc = acc
            self.wait = 0

            predictions = self.model.predict(self.train_x)
            errors = np.abs(predictions - self.train_y) <= self.threshold
            errors = np.sum(errors, axis=1)
            errors = len(errors[errors == predictions.shape[1]])
            is_over_threshold = errors / predictions.shape[0] > self.stop_threshold_1

            if is_over_threshold:
                self.best_weights = self.model.get_weights()
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
