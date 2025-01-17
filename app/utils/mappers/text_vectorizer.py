from typing import Callable

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BaseTextVectorizerMapper:
    def __init__(self):
        self._mapping: dict[str, Callable] = {}

    def register_model(self, name: str, model_class: Callable) -> None:
        self._mapping[name] = model_class

    def get_model(self, name: str, **kwargs) -> Callable:
        model_class = self._mapping.get(name)
        if model_class is None:
            raise ValueError(f"Model '{name}' is not registered.")
        return model_class(**kwargs)


class TextVectorizerMapper(BaseTextVectorizerMapper):
    def __init__(self):
        super().__init__()

        # Text Vectorization Algorithms
        self.register_model("cv", CountVectorizer)
        self.register_model("tfidf", TfidfVectorizer)
