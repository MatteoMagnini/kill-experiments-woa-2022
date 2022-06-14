from typing import Iterable
from resources.data.splice_junction import PATH as SPLICE_JUNCTION_PATH, get_indices as get_spice_junction_indices, \
    FEATURES
import pandas as pd
import re


SPLICE_JUNCTION_INDICES = get_spice_junction_indices()


def get_splice_junction_data(filename: str) -> pd.DataFrame:
    return _get_data(str(SPLICE_JUNCTION_PATH / filename) + '.txt')


def data_to_int(data: pd.DataFrame, mapping: dict[str: int]) -> pd.DataFrame:
    return data.applymap(lambda x: mapping[x] if x in mapping.keys() else x)


def _get_data(file: str) -> pd.DataFrame:
    x = []
    with open(file) as f:
        for row in f:
            row = re.sub('\n', '', row)
            label, _, features = row.split(',')
            features = list(f for f in features.lower())
            features.append(label.lower())
            x.append(features)
    return pd.DataFrame(x)


def get_binary_data(data: pd.DataFrame, mapping: dict[str: set[str]]) -> pd.DataFrame:
    sub_features = sorted(_get_values(mapping))
    results = []
    for _, row in data.iterrows():
        row_result = []
        for value in row:
            positive_features = mapping[value]
            for feature in sub_features:
                row_result.append(1 if feature in positive_features else 0)
        results.append(row_result)
    return pd.DataFrame(results, dtype=int)


def _get_values(mapping: dict[str: set[str]]) -> Iterable[str]:
    result = set()
    for values_set in mapping.values():
        for value in values_set:
            result.add(value)
    return result


def get_splice_junction_feature_mapping(variable_indices: list[int] = SPLICE_JUNCTION_INDICES) -> dict[str: int]:
    return _get_feature_mapping(variable_indices)


def get_splice_junction_extended_feature_mapping(features: list[str] = FEATURES,
                                                 variable_indices: list[int] = SPLICE_JUNCTION_INDICES
                                                 ) -> dict[str: int]:
    return _get_extended_feature_mapping(features, variable_indices)


def _get_feature_mapping(variable_indices: list[int]) -> dict[str: int]:
    return {'X' + ('_' if j < 0 else '') + str(abs(j)): i for i, j in enumerate(variable_indices)}


def _get_extended_feature_mapping(features: list[str], variable_indices: list[int]) -> dict[str: int]:
    result = {'X' + ('_' if j < 0 else '') + str(abs(j)) + f: k + i * len(features)
            for i, j in enumerate(variable_indices) for k, f in enumerate(features)}
    return result
