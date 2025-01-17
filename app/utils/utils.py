import os
import pickle

import pandas as pd


def check_missing_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found in DataFrame: {missing_columns}")


def check_column_types(
    df: pd.DataFrame, columns: list[str], expected_type: str | type
) -> None:
    for column in columns:
        check_missing_columns(df, columns=[column])
        if expected_type == "string" and not pd.api.types.is_string_dtype(df[column]):
            raise TypeError(
                f"Column '{column}' must contain string values, found {df[column].dtype}."
            )
        elif expected_type == "numeric" and not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise TypeError(
                f"Column '{column}' must contain numeric values, found {df[column].dtype}."
            )
        elif isinstance(expected_type, type) and not isinstance(
            df[column].dtype, expected_type
        ):
            raise TypeError(
                f"Column '{column}' must contain values of type {expected_type}, found {df[column].dtype}."
            )


def save_object(obj: object, path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_object(path: str) -> object:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"
