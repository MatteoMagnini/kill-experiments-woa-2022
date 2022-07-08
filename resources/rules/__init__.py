import csv
from pathlib import Path


PATH = Path(__file__).parents[0]


def get_rules(rule_name: str = "kb") -> list[str]:
    result = []
    with open(str(PATH / rule_name) + '.txt', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return result


FEATURE_MAPPING = {
        'S1': 0,
        'R1': 1,
        'S2': 2,
        'R2': 3,
        'S3': 4,
        'R3': 5,
        'S4': 6,
        'R4': 7,
        'S5': 8,
        'R5': 9
    }
CLASS_MAPPING = {
        'nothing': 0,
        'pair': 1,
        'two': 2,
        'three': 3,
        'straight': 4,
        'flush': 5,
        'full': 6,
        'four': 7,
        'straight_flush': 8,
        'royal_flush': 9
    }