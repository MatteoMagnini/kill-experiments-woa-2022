import csv
from pathlib import Path
from resources.rules.poker import PATH as POKER_PATH


PATH = Path(__file__).parents[0]


def get_rules(rule_name: str = "kb") -> list[str]:
    result = []
    with open(str(POKER_PATH / rule_name) + '.txt', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return result
