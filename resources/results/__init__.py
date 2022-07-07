from pathlib import Path
from typing import Iterable, List
import pandas as pd
import numpy as np
from numpy import ndarray, apply_along_axis, zeros

PATH = Path(__file__).parents[0]


def get_results(filename: str) -> pd.DataFrame:
    return pd.read_csv(str(PATH / filename) + '.csv', sep=';')


def sum_confusion_matrix(matrices: Iterable[ndarray]) -> ndarray:
    dim = list(matrices)[0].shape[0]
    new_cm = zeros((dim, dim))
    for matrix in matrices:
        new_cm += matrix
    return new_cm


def single_class_accuracies(matrix: ndarray) -> List[float]:
    accuracies = []
    class_cardinalities = apply_along_axis(lambda x: sum(x), axis=1, arr=matrix)
    for i in range(matrix.shape[0]):
        accuracies.append(matrix[i, i]/class_cardinalities[i])
    return accuracies


def f1s(matrix: ndarray) -> tuple[List[float], List[float]]:
    dim = matrix.shape[0]
    total = np.sum(matrix)
    partial_f1s = []
    supports = []
    for d in range(dim):
        tp = matrix[d, d]
        fp = sum(matrix[d, :]) - tp
        fn = sum(matrix[:, d]) - tp
        f1 = tp / (tp + (fp + fn) / 2)
        support = (tp + fp) / total
        partial_f1s.append(f1)
        supports.append(support)
    return partial_f1s, supports


def macro_f1(matrix: ndarray) -> float:
    dim = matrix.shape[0]
    partial_f1s, _ = f1s(matrix)
    return sum(partial_f1s)/dim


def weighted_f1(matrix: ndarray) -> float:
    partial_f1s, supports = f1s(matrix)
    return sum([i * j for i, j in zip(partial_f1s, supports)])


def accuracy(matrix: ndarray) -> float:
    dim = matrix.shape[0]
    tp, total = 0, 0
    for d in range(dim):
        tp += matrix[d, d]
        total += sum(matrix[d, :])
    return tp/total
