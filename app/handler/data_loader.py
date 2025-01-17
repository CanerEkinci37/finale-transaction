import os

import pandas as pd

from ..utils import utils


class DataLoader:
    def __init__(self, config: dict) -> None:
        self.config = config

    def load_data_from_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def load_data_from_json(self, path: str) -> pd.DataFrame:
        df = pd.read_json(path)
        return df

    def validate_data(self, path: str, columns: list[str]) -> pd.DataFrame:
        file_extension = os.path.splitext(path)[1].lower()
        if file_extension == ".csv":
            df = self.load_data_from_csv(path)
        elif file_extension == ".json":
            df = self.load_data_from_json(path)
        utils.check_missing_columns(df, columns=columns)
        df_copy = df[columns].copy()
        return df_copy
