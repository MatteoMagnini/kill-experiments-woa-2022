from typing import Iterable
import pandas as pd
import tensorflow as tf
from psyki.ski import Injector, Formula
from psyki.ski.injectors import LambdaLayer
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Input, Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense
from resources.execution import Conditions
import datetime
from resources.results import PATH


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
        pd.DataFrame(cm).to_csv(str(PATH / ("confusion_matrix" + str(i) + ".csv")))
        print("\nEvaluation time = " + str(datetime.datetime.now() - now))

        losses.append(loss)
        accuracies.append(acc)
        confusion_matrices.append(cm)
        print("\n\n" + 50 * "-" + " " + str(i+1) + " " + 50 * "-" + "\n\n")
        del predictor
    return pd.DataFrame({'loss': losses, 'acc': accuracies})
