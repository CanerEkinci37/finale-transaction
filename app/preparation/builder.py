import numpy as np
import pandas as pd

from ..utils import utils


class BaseBuilder:
    def __init__(self, config: dict) -> None:
        self.config = config

    def add_column(
        self,
        df: pd.DataFrame,
        column: str,
        value: list[object] | None,
    ) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[column] = value
        return df_copy

    def drop_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        utils.check_missing_columns(df, columns=columns)
        df_copy = df.copy()
        df_copy = df_copy.drop(columns=columns, axis=1)
        return df_copy

    def drop_rows(self, df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
        if not all(0 <= idx < len(df) for idx in indices):
            raise IndexError("One or more indices are out of range.")
        df_copy = df.copy()
        df_copy = df_copy.drop(indices, axis=0)
        return df_copy

    def drop_nan_values(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        utils.check_missing_columns(df, columns=columns)
        df_copy = df.copy()
        df_copy = df_copy.dropna(subset=columns)
        return df_copy

    def fill_missing_values_with_zero(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        utils.check_missing_columns(df, columns=columns)
        df_copy = df.copy()
        df_copy[columns] = df_copy[columns].fillna(0)
        return df_copy

    def fill_missing_values_with_missing(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        utils.check_missing_columns(df, columns=columns)
        df_copy = df.copy()
        df_copy[columns] = df_copy[columns].fillna("missing")
        return df_copy

    def concatenate_data(
        self, df_1: pd.DataFrame, df_2: pd.DataFrame, axis: int = 0
    ) -> pd.DataFrame:
        if axis not in [0, 1]:
            raise ValueError("`axis` must be 0 (rows) or 1 (columns).")
        concat_df = pd.concat([df_1, df_2], axis=axis, ignore_index=(axis == 0))
        return concat_df

    def merge_dataset(
        self,
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        left_column: str,
        right_column: str,
        how: str = "inner",
    ) -> pd.DataFrame:
        utils.check_missing_columns(df_1, columns=[left_column])
        utils.check_missing_columns(df_2, columns=[right_column])
        if how not in ["left", "right", "outer", "inner"]:
            raise ValueError(
                f"Invalid merge type: {how}. Must be one of ['left', 'right', 'outer', 'inner']."
            )
        df1_copy = self.drop_nan_values(df_1, [left_column])
        df2_copy = self.drop_nan_values(df_2, [right_column])
        merge_df = pd.merge(
            left=df1_copy,
            right=df2_copy,
            left_on=left_column,
            right_on=right_column,
            how=how,
        )
        return merge_df


class TrainingBuilder(BaseBuilder):
    def __init__(self, config):
        super().__init__(config)

    def get_sample(
        self,
        df: pd.DataFrame,
        amount: int | None = None,
        frac: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        if amount is None and frac is None:
            raise ValueError("Provide either `amount` or `frac`, not both.")
        if amount is not None and frac is not None:
            raise ValueError("Cannot provide both `amount` and `frac` simultaneously.")
        df_copy = df.copy()
        if amount is not None:
            return df_copy.sample(n=amount, random_state=seed)

        return df_copy.sample(frac=frac, random_state=seed)

    def shuffle_columns(
        self, df: pd.DataFrame, columns: list[str], seed: int = 42
    ) -> pd.DataFrame:
        np.random.seed(seed)
        df_copy = df.reset_index(drop=True).copy()

        shuffled_indices = np.random.permutation(df_copy.index)
        shuffled_values = df_copy.loc[shuffled_indices, columns].reset_index(drop=True)
        df_copy[columns] = shuffled_values

        return df_copy, shuffled_indices


class PredictingBuilder(BaseBuilder):
    pass
