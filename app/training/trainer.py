import os
import pickle

import numpy as np

from ..utils import utils
from ..utils.mappers.ml_model import ClassifierMapper, RegressorMapper

classifier_mapper = ClassifierMapper()
regressor_mapper = RegressorMapper()


class Trainer:
    VALID_TASKS = {"classifying", "regression"}

    def __init__(self, config: dict, task: str, algorithm: str):
        self.config = config
        self.model = self.__initialize_model(task=task, algorithm=algorithm)

    def __initialize_model(self, task, algorithm):
        if task not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task type: '{task}'. Must be one of {self.VALID_TASKS}"
            )
        elif task == "classifying":
            return classifier_mapper.get_model(name=algorithm)
        return regressor_mapper.get_model(name=algorithm)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)

    def save(self, path: str) -> None:
        utils.save_object(self.model, path)
