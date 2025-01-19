import logging

import numpy as np
import pandas as pd

from .handler.data_loader import DataLoader
from .prediction.predictor import Predictor
from .preparation.builder import PredictingBuilder
from .preparation.cleaner import BaseCleaner
from .preparation.extractor import PredictingExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start(config: dict, test_data: list[dict]):
    logger.info("Starting prediction process")
    # Loading Partner Data
    loader = DataLoader(config)
    partner_df = loader.validate_data(
        config.get("dataset_path", {}).get("partner"),
        columns=config.get("selected_columns", {}).get("partner", []),
    )
    logger.info(f"Partner data loaded with shape: {partner_df.shape}")

    # Convert Empty Values to NaN
    cleaner = BaseCleaner(config)
    partner_df = cleaner.replace_values(
        partner_df, target_value="", replace_with=np.nan
    )

    # Converting Numeric Columns
    numeric_columns = config.get("columns_to_modify", {}).get("numeric_columns", {})
    partner_df = cleaner.convert_to_numeric(
        partner_df, columns=numeric_columns.get("partner", [])
    )

    # Clean Text
    text_columns = config.get("columns_to_modify", {}).get("text_columns", {})
    partner_df = cleaner.clean_text(partner_df, columns=text_columns.get("partner", []))

    extractor = PredictingExtractor(config)

    # One Hot Encoding
    ohe_columns = config.get("columns_to_modify", {}).get("ohe_columns", {})

    partner_df_ohe = ohe_columns.get("partner", {})
    partner_df_ohe_columns = partner_df_ohe.get("columns", [])
    partner_df_ohe_path = partner_df_ohe.get("save_load_path")

    partner_df = extractor.apply_one_hot_encoding(
        partner_df, columns=partner_df_ohe_columns, load_paths=partner_df_ohe_path
    )

    # Label Encoding
    le_columns = config.get("columns_to_modify", {}).get("le_columns", {})

    partner_df_le = le_columns.get("partner", {})
    partner_df_le_columns = partner_df_le.get("columns", [])
    partner_df_le_path = partner_df_le.get("save_load_path")

    partner_df = extractor.apply_label_encoding(
        partner_df, columns=partner_df_le_columns, load_paths=partner_df_le_path
    )

    # Text Vectorize
    text_vectorize = config.get("columns_to_extract", {}).get("text_vectorize", {})
    partner_df_text_vectorize = text_vectorize.get("partner", {})
    partner_df_text_vectorize_columns = partner_df_text_vectorize.get("columns", [])
    partner_df_text_vectorize_path = partner_df_text_vectorize.get("save_load_path", [])

    partner_df = extractor.apply_text_vectorization(
        partner_df,
        columns=partner_df_text_vectorize_columns,
        load_paths=partner_df_text_vectorize_path,
    )

    # Text Similarity
    text_similarity = config.get("columns_to_extract", {}).get("text_similarity", {})

    partner_df_text_similarity = text_similarity.get("transaction", {})
    partner_df_text_similarity_columns = partner_df_text_similarity.get("columns", [])
    partner_df_text_similarity_methods = partner_df_text_similarity.get("methods", [])

    partner_df = extractor.get_similarity_score(
        partner_df,
        columns=partner_df_text_similarity_columns,
        algorithms=partner_df_text_similarity_methods,
    )

    # Drop Columns
    selected_columns = config.get("selected_columns", {})
    transaction_selected = selected_columns.get("transaction", [])

    # Create a DataFrame for each test data element
    test_df_list = []
    for data_idx, data in enumerate(test_data):
        test_df = pd.DataFrame([data])
        filtered_test_columns = [
            col
            for col in transaction_selected
            if col in test_df.columns and col not in ["lifnr", "kunnr"]
        ]
        test_df = test_df[filtered_test_columns]
        test_df = cleaner.replace_values(test_df, target_value="", replace_with=np.nan)

        # Converting Numeric Columns
        test_df["bukrs"] = test_df["bukrs"].astype(int)
        test_df["amount"] = test_df["amount"].astype(float)

        # Clean Text
        test_df = cleaner.clean_text(
            test_df, columns=text_columns.get("transaction", [])
        )

        # One Hot Encoding
        test_df_ohe = ohe_columns.get("transaction", {})
        test_df_ohe_columns = test_df_ohe.get("columns", [])
        test_df_ohe_path = test_df_ohe.get("save_load_path")

        test_df = extractor.apply_one_hot_encoding(
            test_df, columns=test_df_ohe_columns, load_paths=test_df_ohe_path
        )

        # Label Encoding
        test_df_le = le_columns.get("transaction", {})
        test_df_le_columns = test_df_le.get("columns", [])
        test_df_le_path = test_df_le.get("save_load_path")

        test_df = extractor.apply_label_encoding(
            test_df, columns=test_df_le_columns, load_paths=test_df_le_path
        )

        # Text Vectorize
        test_df_text_vectorize = text_vectorize.get("transaction", {})
        test_df_text_vectorize_columns = test_df_text_vectorize.get("columns", [])
        test_df_text_vectorize_path = test_df_text_vectorize.get("save_load_path", [])

        test_df = extractor.apply_text_vectorization(
            test_df,
            columns=test_df_text_vectorize_columns,
            load_paths=test_df_text_vectorize_path,
        )

        # Text Similarity
        test_df_text_similarity = text_similarity.get("transaction", {})
        test_df_text_similarity_columns = test_df_text_similarity.get("columns", [])
        test_df_text_similarity_methods = test_df_text_similarity.get("methods", [])

        test_df = extractor.get_similarity_score(
            test_df,
            columns=test_df_text_similarity_columns,
            algorithms=test_df_text_similarity_methods,
        )

        # Ratio Classify Function
        builder = PredictingBuilder(config)
        test_deleted_columns = ["description1", "cleaned_description1"]
        save_load = config.get("columns_to_train", {}).get("save_load", {})

        def predict_ratio_class(df: pd.DataFrame):
            df_copy = df.copy()
            df_copy = builder.drop_columns(df_copy, columns=test_deleted_columns)
            predictor = Predictor(path=save_load.get("classify_ratio"))
            return predictor.predict(df_copy.values)

        ratio_pred = predict_ratio_class(test_df)

        # Lifnr, Kunnr Prediction Function
        def predict_lifnr_kunnr(
            test_df: pd.DataFrame, partner_df: pd.DataFrame, ratio_pred: np.ndarray
        ):
            logger.info("Starting LIFNR/KUNNR prediction")
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
            logger.info(f"Merged DataFrame shape: {merged_df.shape}")

            merged_text_similarity = text_similarity.get("merge", {})
            merged_text_similarity_columns = merged_text_similarity.get("columns", [])
            merged_text_similarity_methods = merged_text_similarity.get("methods", [])

            merged_df = extractor.get_similarity_score(
                merged_df,
                columns=merged_text_similarity_columns,
                algorithms=merged_text_similarity_methods,
            )
            logger.info("Applied text similarity scoring on merged_df")

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
            logger.info(f"Dropped columns from merged_df, new shape: {merged_df.shape}")

            predictor = Predictor(save_load.get(task_name))
            y_pred = predictor.predict(merged_df.values)
            logger.info(f"Prediction completed for {task_name}")

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
            logger.info(f"Top predictions: {pred_results}")
            return pred_results

        # Add result for each test dataframe
        test_df_list.append(
            predict_lifnr_kunnr(test_df, partner_df, ratio_pred=ratio_pred)
        )

    logger.info("Prediction process completed")
    return test_df_list  # Return a list of predictions for each test data
