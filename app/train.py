import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.handler.data_loader import DataLoader
from app.preparation.builder import TrainingBuilder
from app.preparation.cleaner import BaseCleaner
from app.preparation.extractor import TrainingExtractor
from app.training.trainer import Trainer

logger = logging.getLogger(__name__)


def start(config: dict, models: list[str]):
    # Dataset Reading
    loader = DataLoader(config)
    transactions = loader.validate_data(
        path=config.get("dataset_path", {}).get("transaction", ""),
        columns=config.get("selected_columns", {}).get("transaction", []),
    )
    partners = loader.validate_data(
        path=config.get("dataset_path", {}).get("partner", ""),
        columns=config.get("selected_columns", {}).get("partner", []),
    )

    # Convert Empty values to NaN
    cleaner = BaseCleaner(config)
    transactions = cleaner.replace_values(
        transactions, target_value="", replace_with=np.nan
    )
    partners = cleaner.replace_values(partners, target_value="", replace_with=np.nan)

    # Converting to Numeric Value
    numeric_columns = config.get("columns_to_modify", {}).get("numeric_columns", {})
    transactions = cleaner.convert_to_numeric(
        transactions, columns=numeric_columns.get("transaction", [])
    )
    partners = cleaner.convert_to_numeric(
        partners, columns=numeric_columns.get("partner", [])
    )

    # Text Cleaning
    text_columns = config.get("columns_to_modify", {}).get("text_columns", {})
    transactions = cleaner.clean_text(
        transactions, columns=text_columns.get("transaction", [])
    )
    partners = cleaner.clean_text(partners, columns=text_columns.get("partner", []))

    # One Hot Encoding
    extractor = TrainingExtractor(config)
    ohe_columns = config.get("columns_to_modify", {}).get("ohe_columns", {})

    transaction_ohe = ohe_columns.get("transaction", {})
    transaction_ohe_columns = transaction_ohe.get("columns", [])
    transaction_ohe_path = transaction_ohe.get("save_load_path")
    transactions = extractor.apply_one_hot_encoding(
        transactions, columns=transaction_ohe_columns, save_paths=transaction_ohe_path
    )

    partner_ohe = ohe_columns.get("partner", {})
    partner_ohe_columns = partner_ohe.get("columns", [])
    partner_ohe_path = partner_ohe.get("save_load_path")
    partners = extractor.apply_one_hot_encoding(
        partners, columns=partner_ohe_columns, save_paths=partner_ohe_path
    )

    # Label Encoding
    le_columns = config.get("columns_to_modify", {}).get("le_columns", {})

    transaction_le = le_columns.get("transaction", {})
    transaction_le_columns = transaction_le.get("columns", [])
    transaction_le_path = transaction_le.get("save_load_path")
    transactions = extractor.apply_one_hot_encoding(
        transactions, columns=transaction_le_columns, save_paths=transaction_le_path
    )

    partner_le = le_columns.get("partner", {})
    partner_le_columns = partner_le.get("columns", [])
    partner_le_path = partner_le.get("save_load_path")
    partners = extractor.apply_one_hot_encoding(
        partners, columns=partner_le_columns, save_paths=partner_le_path
    )

    # Text Vectorizing
    text_vectorize = config.get("columns_to_extract", {}).get("text_vectorize", {})

    transaction_text_vectorize = text_vectorize.get("transaction", {})
    transaction_text_vectorize_columns = transaction_text_vectorize.get("columns", [])
    transaction_text_vectorize_method = transaction_text_vectorize.get("method")
    transaction_text_vectorize_path = transaction_text_vectorize.get(
        "save_load_path", []
    )
    transactions = extractor.apply_text_vectorization(
        transactions,
        columns=transaction_text_vectorize_columns,
        algorithm=transaction_text_vectorize_method,
        save_paths=transaction_text_vectorize_path,
    )

    partner_text_vectorize = text_vectorize.get("partner", {})
    partner_text_vectorize_columns = partner_text_vectorize.get("columns", [])
    partner_text_vectorize_method = partner_text_vectorize.get("method")
    partner_text_vectorize_path = partner_text_vectorize.get("save_load_path", [])
    partners = extractor.apply_text_vectorization(
        partners,
        columns=partner_text_vectorize_columns,
        algorithm=partner_text_vectorize_method,
        save_paths=partner_text_vectorize_path,
    )

    # Text Similarity
    text_similarity = config.get("columns_to_extract", {}).get("text_similarity", {})

    transaction_text_similarity = text_similarity.get("transaction", {})
    transaction_text_similarity_columns = transaction_text_similarity.get("columns", [])
    transaction_text_similarity_methods = transaction_text_similarity.get("methods", [])
    transactions = extractor.get_similarity_score(
        transactions,
        columns=transaction_text_similarity_columns,
        algorithms=transaction_text_similarity_methods,
    )

    partner_text_similarity = text_similarity.get("transaction", {})
    partner_text_similarity_columns = partner_text_similarity.get("columns", [])
    partner_text_similarity_methods = partner_text_similarity.get("methods", [])
    partners = extractor.get_similarity_score(
        partners,
        columns=partner_text_similarity_columns,
        algorithms=partner_text_similarity_methods,
    )

    # Training Ratio CLassifier
    builder = TrainingBuilder(config)
    filled_columns = config.get("columns_to_build", {}).get("filled_columns", {})
    deleted_columns = config.get("columns_to_build", {}).get("deleted_columns", {})

    def train_ratio_classifer(df):
        df_copy = df.copy()

        df_filled = filled_columns.get("classify_ratio", {})
        df_filled_zero = df_filled.get("zero", [])
        df_filled_missing = df_filled.get("missing", [])

        df_copy = builder.fill_missing_values_with_zero(df_copy, columns=df_filled_zero)
        df_copy = builder.fill_missing_values_with_missing(
            df_copy, columns=df_filled_missing
        )

        df_copy["Ratio"] = df_copy["kunnr"].apply(lambda x: 0 if x == 0 else 1)

        df_deleted = deleted_columns.get("classify_ratio", [])

        df_copy = builder.drop_columns(df_copy, columns=df_deleted)
        X = builder.drop_columns(df_copy, columns=["Ratio"])
        y = df_copy["Ratio"]

        # Split to Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        task_type = config.get("columns_to_train", {}).get("task_type", {})
        df_task_type = task_type.get("classify_ratio")

        algorithm = config.get("columns_to_train", {}).get("algorithm", {})
        df_algorithm = algorithm.get("classify_ratio")

        trainer = Trainer(config, task=df_task_type, algorithm=df_algorithm)
        trainer.train(X_train, y_train)

        # Get Metrics
        y_pred = trainer.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "precision": precision_score(y_test, y_pred, pos_label=1),
            "f1": f1_score(y_test, y_pred, pos_label=1),
        }

        save_load = config.get("columns_to_train", {}).get("save_load", {})
        df_save_load = save_load.get("classify_ratio")

        trainer.save(df_save_load)
        logger.info("Ratio Classifier Model is trained")

        return metrics

    # Training Lifnr, Kunnr Regressor
    merged_columns = config.get("columns_to_build", {}).get("merged_columns", {})

    def train_lifnr_kunnr_regressor(transaction_df, partner_df, task_name: str):
        transaction_copy = transaction_df.copy()
        partner_copy = partner_df.copy()

        left_key = merged_columns.get("left_key", [])
        right_key = merged_columns.get("right_key", [])

        merged_df1 = builder.merge_dataset(
            df_1=transaction_copy,
            df_2=partner_copy,
            left_column=left_key[0],
            right_column=right_key[0],
        )

        merged_df2 = builder.merge_dataset(
            df_1=transaction_copy,
            df_2=partner_copy,
            left_column=left_key[1],
            right_column=right_key[1],
        )

        task_filled = filled_columns.get(task_name)

        real_df = builder.concatenate_data(merged_df1, merged_df2, axis=0)
        real_df = builder.fill_missing_values_with_zero(
            real_df, columns=task_filled.get("zero", [])
        )
        real_df = builder.fill_missing_values_with_missing(
            real_df, columns=task_filled.get("missing", [])
        )

        task_text_similarity = text_similarity.get("merge", {})
        real_df = extractor.get_similarity_score(
            real_df,
            columns=task_text_similarity.get("columns", []),
            algorithms=task_text_similarity.get("methods", []),
        )

        real_df["Ratio"] = real_df[task_name.split("_")[-1].upper()].apply(
            lambda x: 0 if x == 0 else 1
        )

        ratio_1 = real_df[real_df["Ratio"] == 1]
        sampled_df = builder.get_sample(ratio_1, frac=1)
        shuffled_df, shuffled_indices = builder.shuffle_columns(
            sampled_df,
            columns=config.get("columns_to_build", {}).get("shuffled_columns", []),
        )

        shuffled_ratios = [
            1 if a == b else 0 for a, b in zip(shuffled_df.index, shuffled_indices)
        ]

        shuffled_df = builder.add_column(
            shuffled_df, column="Ratio", value=shuffled_ratios
        )

        final_df = builder.concatenate_data(real_df, shuffled_df, axis=0)
        final_df = builder.drop_columns(
            final_df, columns=deleted_columns.get(task_name, [])
        )

        X = builder.drop_columns(final_df, columns=["Ratio"])
        y = final_df["Ratio"]

        # Split to Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        task_train = config.get("columns_to_train", {})
        task_type = task_train.get("task_type", {}).get(task_name)
        task_algorithm = task_train.get("algorithm", {}).get(task_name)
        task_save_load = task_train.get("save_load", {}).get(task_name)

        trainer = Trainer(config, task=task_type, algorithm=task_algorithm)
        trainer.train(X_train.values, y_train)

        # Get Metrics
        y_pred = trainer.predict(X_test.values)
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        trainer.save(task_save_load)

        logger.info(f"{task_name} regressor model training complete.")

        return metrics

    available_models = {
        "ratio": lambda: train_ratio_classifer(transactions),
        "lifnr": lambda: train_lifnr_kunnr_regressor(
            transactions, partners, "predict_lifnr"
        ),
        "kunnr": lambda: train_lifnr_kunnr_regressor(
            transactions, partners, "predict_kunnr"
        ),
    }

    # If no models are specified, train all models
    if models is None:
        models = list(available_models.keys())

    # Validate the specified models
    invalid_models = [model for model in models if model not in available_models]
    if invalid_models:
        raise ValueError(
            f"Invalid models specified: {invalid_models}. Available models: {list(available_models.keys())}"
        )

    logger.info(f"Starting training for models: {models}")

    # Run the training workflows for specified models
    model_metrics = {}
    for model in models:
        logger.info(f"Training model: {model}")
        metrics = available_models[model]()
        model_metrics[model] = metrics

    logger.info(f"Training completed for models: {models}")
    return model_metrics
