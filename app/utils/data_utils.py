import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ..handler.data_loader import DataLoader
from ..preparation.cleaner import BaseCleaner
from ..preparation.extractor import PredictingExtractor, TrainingExtractor


class DataUtils:
    def __init__(self, config: dict, mode: str = "train"):
        self.config = config
        self.loader = DataLoader(config)
        self.cleaner = BaseCleaner(config)
        self.extractor = (
            TrainingExtractor(config)
            if mode == "train"
            else PredictingExtractor(config)
        )
        self.mode = mode

    def _validate_data(self, dataset_name: str) -> pd.DataFrame:
        path = self.config.get("dataset_path", {}).get(dataset_name, "")
        selected_columns = self.config.get("selected_columns", {}).get(dataset_name, [])
        return self.loader.validate_data(path, columns=selected_columns)

    def _replace_empty_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        return self.cleaner.replace_values(
            df_copy, target_value="", replace_with=np.nan
        )

    def _convert_to_numeric(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df_copy = df.copy()
        numeric_columns = self.config.get("columns_to_modify", {}).get(
            "numeric_columns", {}
        )
        return self.cleaner.convert_to_numeric(
            df_copy, columns=numeric_columns.get(dataset_name, [])
        )

    def _text_clean(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df_copy = df.copy()
        text_columns = self.config.get("columns_to_modify", {}).get("text_columns", {})
        return self.cleaner.clean_text(
            df_copy, columns=text_columns.get(dataset_name, [])
        )

    def _get_onehot_encode(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df_copy = df.copy()
        ohe_columns = self.config.get("columns_to_modify", {}).get("ohe_columns", {})

        df_ohe = ohe_columns.get(dataset_name, {})
        df_ohe_columns = df_ohe.get("columns", [])
        df_ohe_path = df_ohe.get("save_load_path")
        return self.extractor.apply_one_hot_encoding(
            df_copy, df_ohe_columns, df_ohe_path
        )

    def _get_label_encode(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df_copy = df.copy()
        le_columns = self.config.get("columns_to_modify", {}).get("le_columns", {})

        df_le = le_columns.get(dataset_name, {})
        df_le_columns = df_le.get("columns", [])
        df_le_path = df_le.get("save_load_path")
        return self.extractor.apply_one_hot_encoding(df_copy, df_le_columns, df_le_path)

    def _get_text_vectorization(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        df_copy = df.copy()
        text_vectorize = self.config.get("columns_to_extract", {}).get(
            "text_vectorize", {}
        )

        df_text_vectorize = text_vectorize.get(dataset_name, {})
        df_text_vectorize_columns = df_text_vectorize.get("columns", [])
        df_text_vectorize_path = df_text_vectorize.get("save_load_path", [])

        if self.mode == "train":
            df_text_vectorize_method = df_text_vectorize.get("method")
            return self.extractor.apply_text_vectorization(
                df_copy,
                df_text_vectorize_columns,
                df_text_vectorize_method,
                df_text_vectorize_path,
            )
        return self.extractor.apply_text_vectorization(
            df_copy, df_text_vectorize_columns, df_text_vectorize_path
        )

    def _get_text_similarity(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df_copy = df.copy()
        text_similarity = self.config.get("columns_to_extract", {}).get(
            "text_similarity", {}
        )

        df_text_similarity = text_similarity.get(dataset_name, {})
        df_text_similarity_columns = df_text_similarity.get("columns", [])
        df_text_similarity_methods = df_text_similarity.get("methods", [])
        return self.extractor.get_similarity_score(
            df_copy,
            columns=df_text_similarity_columns,
            algorithms=df_text_similarity_methods,
        )

    def process_train_data(self, dataset_name: str) -> pd.DataFrame:
        df = self._validate_data(dataset_name)
        df = self._replace_empty_to_nan(df)
        df = self._convert_to_numeric(df, dataset_name=dataset_name)
        df = self._text_clean(df, dataset_name=dataset_name)
        df = self._get_onehot_encode(df, dataset_name=dataset_name)
        df = self._get_label_encode(df, dataset_name=dataset_name)
        df = self._get_text_vectorization(df, dataset_name=dataset_name)
        return self._get_text_similarity(df, dataset_name=dataset_name)

    def process_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy = self._replace_empty_to_nan(df_copy)
        df_copy["bukrs"] = df_copy["bukrs"].astype(int)
        df_copy["amount"] = df_copy["amount"].astype(float)
        df_copy = self._text_clean(df_copy, dataset_name="transaction")
        df_copy = self._get_onehot_encode(df_copy, dataset_name="transaction")
        df_copy = self._get_label_encode(df_copy, dataset_name="transaction")
        df_copy = self._get_text_vectorization(df_copy, dataset_name="transaction")
        return self._get_text_similarity(df_copy, dataset_name="transaction")

    def calculate_metrics(self, y_true, y_pred, task_type: str):
        if task_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred, pos_label=1),
                "precision": precision_score(y_true, y_pred, pos_label=1),
                "f1": f1_score(y_true, y_pred, pos_label=1),
            }
        elif task_type == "regression":
            return {
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
            }
        else:
            raise ValueError(f"Unknown task type: {task_type}")
