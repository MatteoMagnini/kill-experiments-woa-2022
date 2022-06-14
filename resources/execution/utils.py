from typing import Iterable
import numpy as np
import pandas as pd
from psyki.ski import Injector, Formula
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Input, Model
from tensorflow.keras.metrics import sparse_categorical_crossentropy, sparse_categorical_accuracy
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Dropout
from resources.execution import Conditions


def create_fully_connected_nn_with_dropout(input_size: int = 60*4) -> Model:
    inputs = Input((input_size, ))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(3, 'softmax')(x)
    return Model(inputs, x)


def create_standard_fully_connected_nn(input_size: int, output_size, layers: int, neurons: int, activation: str) -> Model:
    inputs = Input((input_size,))
    x = Dense(neurons, activation=activation)(inputs)
    for _ in range(1, layers):
        x = Dense(neurons, activation=activation)(x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    return Model(inputs, x)


def split(data: pd.DataFrame,
          injector: Injector,
          knowledge: Iterable[Formula],
          use_knowledge: bool = True,
          training_ratio: float = 2/3,
          population_size: int = 1,
          seed: int = 0,
          epochs: int = 100,
          batch_size: int = 32,
          stop: bool = True
          ) -> pd.DataFrame:
    losses, accuracies = [], []
    n, ie, ei = [], [], []
    for i in range(0, population_size):
        seed = seed + i
        set_seed(seed)
        predictor = injector.inject(knowledge) if use_knowledge else clone_model(injector.predictor)
        predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train, test = train_test_split(data, train_size=training_ratio, random_state=seed, stratify=data.iloc[:, -1])
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        test_n = test.loc[data[240] == 2]
        test_ie = test.loc[data[240] == 1]
        test_ei = test.loc[data[240] == 0]
        early_stop = Conditions(train_x, train_y) if stop else None
        predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=early_stop)
        predictor.remove_constraints()
        # injector.gamma = - injector.gamma / 2
        # predictor = injector.inject(knowledge) if use_knowledge else clone_model(injector.predictor)
        # predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, acc = predictor.evaluate(test_x, test_y)
        losses.append(loss)
        accuracies.append(acc)
        n_predictions = predictor.predict(test_n.iloc[:, :-1])
        ie_predictions = predictor.predict(test_ie.iloc[:, :-1])
        ei_predictions = predictor.predict(test_ei.iloc[:, :-1])
        n.append(np.mean(sparse_categorical_accuracy(test_n.iloc[:, -1:], n_predictions)))
        ie.append(np.mean(sparse_categorical_accuracy(test_ie.iloc[:, -1:], ie_predictions)))
        ei.append(np.mean(sparse_categorical_accuracy(test_ei.iloc[:, -1:], ei_predictions)))
        del predictor
    return pd.DataFrame({'loss': losses, 'acc': accuracies, 'n_acc': n, 'ei_acc': ei, 'ie_acc': ie})


def k_fold_cross_validation(data: pd.DataFrame,
                            injector: Injector,
                            knowledge: list[Formula],
                            use_knowledge: bool = True,
                            k: int = 10,
                            population_size: int = 30,
                            seed: int = 0,
                            epochs: int = 100,
                            batch_size: int = 32,
                            stop: bool = True
                            ) -> pd.DataFrame:
    losses, accuracies = [], []
    n, ie, ei = [], [], []
    for i in range(0, population_size):
        set_seed(seed + i)
        trained_predictors = []
        train, test = train_test_split(data, train_size=1000, random_state=seed, stratify=data.iloc[:, -1])
        test_n = test.loc[data[240] == 2]
        test_ie = test.loc[data[240] == 1]
        test_ei = test.loc[data[240] == 0]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_indices, test_indices in k_fold.split(train.iloc[:, :-1], train.iloc[:, -1:]):
            predictor = injector.inject(knowledge) if use_knowledge else clone_model(injector.predictor)
            predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            train_x, train_y = train.iloc[train_indices, :-1], train.iloc[train_indices, -1:]
            early_stop = Conditions(train_x, train_y) if stop else None
            predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=early_stop)
            predictor.remove_constraints()
            trained_predictors.append(predictor)
        predictions = np.array([predictor.predict(test_x) for predictor in trained_predictors])
        n_predictions = np.array([predictor.predict(test_n.iloc[:, :-1]) for predictor in trained_predictors])
        ie_predictions = np.array([predictor.predict(test_ie.iloc[:, :-1]) for predictor in trained_predictors])
        ei_predictions = np.array([predictor.predict(test_ei.iloc[:, :-1]) for predictor in trained_predictors])
        predictions = np.sum(predictions, axis=0) / k
        n_predictions = np.sum(n_predictions, axis=0) / k
        ie_predictions = np.sum(ie_predictions, axis=0) / k
        ei_predictions = np.sum(ei_predictions, axis=0) / k
        losses.append(np.mean(sparse_categorical_crossentropy(test_y, predictions)))
        accuracies.append(np.mean(sparse_categorical_accuracy(test_y, predictions)))
        n.append(np.mean(sparse_categorical_accuracy(test_n.iloc[:, -1:], n_predictions)))
        ie.append(np.mean(sparse_categorical_accuracy(test_ie.iloc[:, -1:], ie_predictions)))
        ei.append(np.mean(sparse_categorical_accuracy(test_ei.iloc[:, -1:], ei_predictions)))
        print(i+1)
        print(losses[-1])
        print(accuracies[-1])
        del trained_predictors
    return pd.DataFrame({'loss': losses, 'acc': accuracies, 'n_acc': n, 'ei_acc': ei, 'ie_acc': ie})
