import re

import pandas as pd

from ..constants import should_not_be_in_tfidf_matrix, tr_stopwords


class BaseCleaner:
    def __init__(self, config: dict) -> None:
        self.config = config

    def replace_values(
        self,
        df: pd.DataFrame,
        target_value: object,
        replace_with: object,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        df_copy = df.copy()
        if columns:
            missing_columns = [col for col in columns if col not in df_copy.columns]
            if missing_columns:
                raise KeyError(f"Columns not found in DataFrame: {missing_columns}")
            for col in columns:
                df_copy[col] = df_copy[col].replace(
                    to_replace=target_value, value=replace_with
                )
        else:
            df_copy = df_copy.replace(to_replace=target_value, value=replace_with)
        return df_copy

    def convert_to_numeric(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df_copy = df.copy()
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found in DataFrame: {missing_columns}")
        df_copy[columns] = (
            df_copy[columns]
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
            .astype("Int64")
        )
        return df_copy

    def convert_to_object(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df_copy = df.copy()
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found in DataFrame: {missing_columns}")
        df_copy[columns] = df_copy[columns].astype("object")
        return df_copy

    def _normalize_turkish_characters(self, text: str) -> str:
        turkish_map = {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "Ç": "C",
            "Ğ": "G",
            "İ": "I",
            "Ö": "O",
            "Ş": "S",
            "Ü": "U",
        }
        return "".join(turkish_map.get(c, c) for c in text)

    def _remove_non_alpha_characters(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z ]", "", text)

    def _convert_text_to_lower(self, text: str) -> str:
        return text.lower()

    def _remove_stop_words(self, text: str) -> str:
        words = text.split()
        filtered_tokens = [w for w in words if w not in tr_stopwords]
        return " ".join(filtered_tokens)

    def _remove_specific_words(self, text: str, words: list[str]) -> str:
        tokens = text.split()
        filtered_tokens = [w for w in tokens if w not in words]
        return " ".join(filtered_tokens)

    def _strip_text(self, text: str) -> str:
        return text.strip()

    def _concat_words(self, text: str) -> str:
        return text.replace(" ", "")

    def clean_text(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        df_copy = df.copy()
        missing_columns = [col for col in columns if col not in df_copy.columns]
        if missing_columns:
            raise KeyError(f"Columns not found in DataFrame: {missing_columns}")
        for col in columns:
            df_copy[f"cleaned_{col}"] = (
                df_copy[col]
                .astype(str)
                .apply(self._normalize_turkish_characters)
                .apply(self._remove_non_alpha_characters)
                .apply(self._convert_text_to_lower)
                .apply(self._remove_stop_words)
                .apply(
                    self._remove_specific_words,
                    words=should_not_be_in_tfidf_matrix,
                )
                .apply(self._strip_text)
                # .apply(self._concat_words)
            )
        return df_copy
