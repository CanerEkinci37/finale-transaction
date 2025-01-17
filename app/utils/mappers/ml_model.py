from typing import Protocol, Type

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class MLModelProtocol(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | list[float] = None,
    ) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...


class BaseModelMapper:
    def __init__(self):
        self._mapping: dict[str, Type[MLModelProtocol]] = {}

    def register_model(self, name: str, model_class: Type[MLModelProtocol]):
        self._mapping[name] = model_class

    def get_model(self, name: str, **kwargs):
        model_class = self._mapping.get(name)
        if model_class is None:
            raise ValueError(f"Model '{name}' is not registered.")
        return model_class(**kwargs)


class RegressorMapper(BaseModelMapper):
    def __init__(self):
        super().__init__()

        # Linear models
        self.register_model("lr", LinearRegression)
        self.register_model("ridge", Ridge)
        self.register_model("lasso", Lasso)
        self.register_model("elasticnet", ElasticNet)

        # Tree-based models
        self.register_model("dt", DecisionTreeRegressor)
        self.register_model("rf", RandomForestRegressor)
        self.register_model("gbr", GradientBoostingRegressor)

        # Neighbor-based models
        self.register_model("knn", KNeighborsRegressor)

        # Support Vector Machines
        self.register_model("svm", SVR)

        # Boosting models
        self.register_model("catboost", CatBoostRegressor)
        self.register_model("xgboost", XGBRegressor)
        self.register_model("lgbm", LGBMRegressor)


class ClassifierMapper(BaseModelMapper):
    def __init__(self):
        super().__init__()

        # Linear models
        self.register_model("lr", LogisticRegression)

        # Tree-based models
        self.register_model("dt", DecisionTreeClassifier)
        self.register_model("rf", RandomForestClassifier)
        self.register_model("gbr", GradientBoostingClassifier)

        # Neighbor-based models
        self.register_model("knn", KNeighborsClassifier)

        # Support Vector Machines
        self.register_model("svm", SVC)

        # Boosting models
        self.register_model("catboost", CatBoostClassifier)
        self.register_model("xgboost", XGBClassifier)
        self.register_model("lgbm", LGBMClassifier)
