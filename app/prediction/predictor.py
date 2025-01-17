import pickle

import numpy as np


class Predictor:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
