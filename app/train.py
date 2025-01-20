import logging

from sklearn.model_selection import train_test_split

from app.preparation.builder import TrainingBuilder
from app.training.trainer import Trainer

from .preparation.extractor import TrainingExtractor
from .utils.data_utils import DataUtils

logger = logging.getLogger(__name__)


def start(config: dict, models: list[str]):
    data_utils = DataUtils(config, mode="train")

    transactions = data_utils.process_train_data(dataset_name="transaction")
    partners = data_utils.process_train_data(dataset_name="partner")

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        task_train = config.get("columns_to_train", {})
        task_type = task_train.get("task_type", {}).get("classify_ratio")
        task_algorithm = task_train.get("algorithm", {}).get("classify_ratio")
        task_save = task_train.get("save_load", {}).get("classify_ratio")

        trainer = Trainer(config, task=task_type, algorithm=task_algorithm)
        trainer.train(X_train, y_train)

        y_pred = trainer.predict(X_test)
        trainer.save(task_save)
        logger.info("Ratio Classifier Model is trained")
        return data_utils.calculate_metrics(y_test, y_pred, task_type="classification")

    merged_columns = config.get("columns_to_build", {}).get("merged_columns", {})

    def train_lifnr_kunnr_regressor(transaction_df, partner_df, task_name: str):
        text_similarity = config.get("columns_to_extract", {}).get(
            "text_similarity", {}
        )
        extractor = TrainingExtractor(config)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        task_train = config.get("columns_to_train", {})
        task_type = task_train.get("task_type", {}).get(task_name)
        task_algorithm = task_train.get("algorithm", {}).get(task_name)
        task_save_load = task_train.get("save_load", {}).get(task_name)

        trainer = Trainer(config, task=task_type, algorithm=task_algorithm)
        trainer.train(X_train.values, y_train)

        y_pred = trainer.predict(X_test.values)
        trainer.save(task_save_load)
        logger.info(f"{task_name} regressor model training complete.")
        return data_utils.calculate_metrics(y_test, y_pred, task_type="regression")

    available_models = {
        "ratio": lambda: train_ratio_classifer(transactions),
        "lifnr": lambda: train_lifnr_kunnr_regressor(
            transactions, partners, "predict_lifnr"
        ),
        "kunnr": lambda: train_lifnr_kunnr_regressor(
            transactions, partners, "predict_kunnr"
        ),
    }

    if models is None:
        models = list(available_models.keys())
    invalid_models = [model for model in models if model not in available_models]
    if invalid_models:
        raise ValueError(
            f"Invalid models specified: {invalid_models}. Available models: {list(available_models.keys())}"
        )

    logger.info(f"Starting training for models: {models}")
    model_metrics = {}
    for model in models:
        logger.info(f"Training model: {model}")
        metrics = available_models[model]()
        model_metrics[model] = metrics
    logger.info(f"Training completed for models: {models}")
    return model_metrics
