from pathlib import Path
import pandas as pd

PATH = Path(__file__).parents[0]


def get_results(filename: str) -> pd.DataFrame:
    return pd.read_csv(str(PATH / filename) + '.csv', sep=';')
