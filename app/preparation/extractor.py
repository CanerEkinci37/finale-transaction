import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ..utils import utils
from ..utils.mappers.text_similarity import TextSimilarityMapper
from ..utils.mappers.text_vectorizer import TextVectorizerMapper

text_similarity_mapper = TextSimilarityMapper()
text_vectorizer_mapper = TextVectorizerMapper()


class BaseExtractor:
    def __init__(self, config: dict) -> None:
        self.config = config

    def get_similarity_score(
        self, df: pd.DataFrame, columns: list[list[str, str]], algorithms: list[str]
    ) -> pd.DataFrame:
        df_copy = df.copy()
        for col1, col2 in columns:
            utils.check_missing_columns(df, columns=[col1, col2])
            utils.check_column_types(df, columns=[col1, col2], expected_type="string")
            for algo in algorithms:
                algo_func = text_similarity_mapper.get_method(algo)
                df_copy[f"{algo}_{col1}_{col2}"] = [
                    algo_func(str(val1), str(val2))
                    for val1, val2 in zip(df_copy[col1], df_copy[col2])
                ]
        return df_copy


class TrainingExtractor(BaseExtractor):
    def __init__(self, config: dict):
        super().__init__(config)

    def apply_label_encoding(
        self, df: pd.DataFrame, columns: list[str], save_paths: list[str] = None
    ) -> pd.DataFrame:
        df_copy = df.copy()
        utils.check_missing_columns(df, columns=columns)
        if save_paths and len(save_paths) != len(columns):
            raise ValueError("Number of save paths must match the number of columns")

        for idx, col in enumerate(columns):
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            utils.save_object(le, save_paths[idx])
        return df_copy

    def apply_one_hot_encoding(
        self, df: pd.DataFrame, columns: list[str], save_paths: list[str]
    ) -> pd.DataFrame:
        df_copy = df.copy()
        utils.check_missing_columns(df, columns=columns)
        if save_paths and len(save_paths) != len(columns):
            raise ValueError("Number of save paths must match the number of columns")

        for idx, col in enumerate(columns):
            ohe = OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="ignore"
            )
            encoded_columns = ohe.fit_transform(df_copy[[col]])
            encoded_df = pd.DataFrame(
                encoded_columns, columns=ohe.get_feature_names_out()
            )
            df_copy = pd.concat((df_copy.drop(col, axis=1), encoded_df), axis=1)
            utils.save_object(ohe, save_paths[idx])
        return df_copy

    def apply_text_vectorization(
        self, df: pd.DataFrame, columns: list[str], algorithm: str, save_paths: str
    ):
        utils.check_missing_columns(df, columns=columns)
        utils.check_column_types(df, columns=columns, expected_type="string")
        df_copy = df.copy()
        if save_paths and len(save_paths) != len(columns):
            raise ValueError("Number of save paths must match the number of columns")

        for idx, col in enumerate(columns):
            algo_func = text_vectorizer_mapper.get_model(
                algorithm, max_features=200, token_pattern=r"\b\w{2,}\b"
            )
            vectorized_data = algo_func.fit_transform(df_copy[col])
            feature_names = algo_func.get_feature_names_out()

            vectorized_df = pd.DataFrame(
                vectorized_data.toarray(), columns=feature_names
            )
            df_copy = pd.concat([df_copy, vectorized_df], axis=1)
            utils.save_object(algo_func, save_paths[idx])
        return df_copy


class PredictingExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)

    def apply_label_encoding(
        self, df: pd.DataFrame, columns: list[str], load_paths: list[str] = None
    ) -> pd.DataFrame:
        df_copy = df.copy()
        utils.check_missing_columns(df, columns=columns)
        if load_paths and len(load_paths) != len(columns):
            raise ValueError("Number of save paths must match the number of columns")

        for idx, col in enumerate(columns):
            le = utils.read_object(load_paths[idx])
            df_copy[col] = le.transform(df_copy[col].astype(str))
        return df_copy

    def apply_one_hot_encoding(
        self, df: pd.DataFrame, columns: list[str], load_paths: list[str]
    ) -> pd.DataFrame:
        df_copy = df.copy()
        utils.check_missing_columns(df, columns=columns)
        if load_paths and len(load_paths) != len(columns):
            raise ValueError("Number of save paths must match the number of columns")

        for idx, col in enumerate(columns):
            ohe = utils.read_object(load_paths[idx])
            encoded_columns = ohe.transform(df_copy[[col]])
            encoded_df = pd.DataFrame(
                encoded_columns, columns=ohe.get_feature_names_out()
            )
            df_copy = pd.concat((df_copy.drop(col, axis=1), encoded_df), axis=1)
        return df_copy

    def apply_text_vectorization(
        self, df: pd.DataFrame, columns: list[str], load_paths: list[str]
    ):
        utils.check_missing_columns(df, columns=columns)
        utils.check_column_types(df, columns=columns, expected_type="string")
        df_copy = df.copy()

        for idx, col in enumerate(columns):
            algo_func = utils.read_object(load_paths[idx])
            vectorized_data = algo_func.transform(df_copy[col])
            feature_names = algo_func.get_feature_names_out()

            vectorized_df = pd.DataFrame(
                vectorized_data.toarray(), columns=feature_names
            )
            df_copy = pd.concat([df_copy, vectorized_df], axis=1)
        return df_copy
