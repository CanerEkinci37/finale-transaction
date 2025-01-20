import logging

import numpy as np
import pandas as pd

from .handler.data_loader import DataLoader
from .prediction.predictor import Predictor
from .preparation.builder import PredictingBuilder
from .preparation.cleaner import BaseCleaner
from .preparation.extractor import PredictingExtractor
from .utils.data_utils import DataUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start(config: dict, test_data: list[dict]):
    data_utils = DataUtils(config, mode="test")

    partner_df = data_utils.process_train_data(dataset_name="partner")

    # Drop Columns
    selected_columns = config.get("selected_columns", {})
    transaction_selected = selected_columns.get("transaction", [])

    # Create a DataFrame for each test data element
    test_df_list = []
    for idx, data in enumerate(test_data):
        test_df = pd.DataFrame([data])
        filtered_test_columns = [
            col
            for col in transaction_selected
            if col in test_df.columns and col not in ["lifnr", "kunnr"]
        ]
        test_df = test_df[filtered_test_columns]
        test_df = data_utils.process_test_data(test_df)

        builder = PredictingBuilder(config)
        test_deleted_columns = ["description1", "cleaned_description1"]
        save_load = config.get("columns_to_train", {}).get("save_load", {})

        def predict_ratio_class(df: pd.DataFrame):
            df_copy = df.copy()
            df_copy = builder.drop_columns(df_copy, columns=test_deleted_columns)
            predictor = Predictor(path=save_load.get("classify_ratio"))
            return predictor.predict(df_copy.values)

        ratio_pred = predict_ratio_class(test_df)

        def predict_lifnr_kunnr(
            test_df: pd.DataFrame, partner_df: pd.DataFrame, ratio_pred: np.ndarray
        ):
            extractor = PredictingExtractor(config)
            text_similarity = config.get("columns_to_extract", {}).get(
                "text_similarity", {}
            )

            test_df_copy = test_df.copy()
            partner_df_copy = partner_df.copy()

            partner_df_copy = builder.fill_missing_values_with_zero(
                partner_df, columns=["LIFNR", "KUNNR"]
            )

            predict_column = "KUNNR" if ratio_pred[0] == 1 else "LIFNR"
            filter_column = "LIFNR" if predict_column == "KUNNR" else "KUNNR"

            partner_df_copy = partner_df_copy[
                partner_df_copy[filter_column] == 0
            ].reset_index(drop=True)
            indices = partner_df_copy[predict_column].values
            task_name = f"predict_{predict_column.lower()}"

            merged_df = pd.merge(test_df_copy, partner_df_copy, how="cross")

            merged_text_similarity = text_similarity.get("merge", {})
            merged_text_similarity_columns = merged_text_similarity.get("columns", [])
            merged_text_similarity_methods = merged_text_similarity.get("methods", [])
            merged_df = extractor.get_similarity_score(
                merged_df,
                columns=merged_text_similarity_columns,
                algorithms=merged_text_similarity_methods,
            )

            merged_deleted_columns = [
                "description1",
                "cleaned_description1",
                "PARTNER",
                "cleaned_PARTNER",
                "TAX_NUMBER",
                "IBAN",
                "KUNNR",
                "LIFNR",
            ]

            merged_df = builder.drop_columns(merged_df, columns=merged_deleted_columns)

            predictor = Predictor(save_load.get(task_name))
            y_pred = predictor.predict(merged_df.values)

            top_5_indices = y_pred.argsort()[-20:][::-1]
            top_5_pred = y_pred[top_5_indices]
            top_5_indices_values = indices[top_5_indices]

            pred_results = []
            for a, b in zip(top_5_indices_values, top_5_pred):
                pred_results.append(
                    {
                        filter_column.lower(): "0",
                        predict_column.lower(): str(a),
                        "company_name": partner_df_copy[
                            partner_df_copy[predict_column] == a
                        ]["PARTNER"].values[0],
                        "score": str(b),
                    }
                )
            return pred_results

        test_df_list.append(
            predict_lifnr_kunnr(test_df, partner_df, ratio_pred=ratio_pred)
        )
    return test_df_list
