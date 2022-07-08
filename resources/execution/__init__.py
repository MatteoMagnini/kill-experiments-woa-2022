import datetime
import os
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import tensorflow as tf
from psyki.ski import Injector, Formula
from psyki.ski.injectors import LambdaLayer
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model
from resources.results import PATH as RESULTS_PATH


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
        self.best_weights = None
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        self.best_acc = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        # Second condition
        if self.best_acc >= acc > self.stop_threshold_2:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
        elif acc > self.best_acc:
            self.best_acc = acc
            self.wait = 0
            self.best_weights = self.model.get_weights()

        # First condition
        predictions = self.model.predict(self.train_x)
        errors = np.abs(predictions - self.train_y) <= self.threshold
        errors = np.sum(errors, axis=1)
        errors = len(errors[errors == predictions.shape[1]])
        is_over_threshold = errors / predictions.shape[0] > self.stop_threshold_1

        if is_over_threshold:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def create_standard_fully_connected_nn(input_size: int, output_size, layers: int, neurons: int, activation: str) -> Model:
    inputs = Input((input_size,))
    x = Dense(neurons, activation=activation)(inputs)
    for _ in range(1, layers):
        x = Dense(neurons, activation=activation)(x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    return Model(inputs, x)


def run_experiments(data: pd.DataFrame,
                    injector: Injector,
                    knowledge: Iterable[Formula],
                    test: pd.DataFrame = None,
                    use_knowledge: bool = True,
                    population_size: int = 30,
                    seed: int = 1,
                    epochs: int = 100,
                    batch_size: int = 32,
                    stop: bool = True
                    ) -> pd.DataFrame:
    losses, accuracies, confusion_matrices = [], [], []
    for i in range(0, population_size):
        seed = seed + i
        set_seed(seed)

        now = datetime.datetime.now()
        predictor = injector.inject(knowledge) if use_knowledge else clone_model(injector._predictor)
        print("Injection time = " + str(datetime.datetime.now() - now))

        predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_x, train_y, test_x, test_y = data.iloc[:, :-1], data.iloc[:, -1:], test.iloc[:, :-1], test.iloc[:, -1:]
        early_stop = Conditions(train_x, train_y) if stop else None

        now = datetime.datetime.now()
        predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, callbacks=early_stop)
        print("Training time = " + str(datetime.datetime.now() - now) + "\n")

        if isinstance(predictor, LambdaLayer.ConstrainedModel):
            predictor = predictor.remove_constraints()
        predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        now = datetime.datetime.now()
        loss, acc = predictor.evaluate(test_x, test_y)
        cm = tf.math.confusion_matrix(test_y.iloc[:, 0], pd.DataFrame(predictor.predict(test_x)).idxmax(axis=1))
        print(cm)
        file_name = ('knowledge' if use_knowledge else 'classic') + os.sep + "confusion_matrix" + str(i) + ".csv"
        pd.DataFrame(cm).to_csv(str(RESULTS_PATH / file_name))
        print("\nEvaluation time = " + str(datetime.datetime.now() - now))

        losses.append(loss)
        accuracies.append(acc)
        confusion_matrices.append(cm)
        print("\n\n" + 50 * "-" + " " + str(i+1) + " " + 50 * "-" + "\n\n")
        del predictor
    return pd.DataFrame({'loss': losses, 'acc': accuracies})
