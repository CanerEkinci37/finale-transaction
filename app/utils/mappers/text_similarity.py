import difflib
from typing import Callable

import jellyfish
import pylcs
import strsimpy
from fuzzywuzzy import fuzz


class BaseExtractorAlgorithm:
    def _calculate_levenshtein(self, s1: str, s2: str) -> int:
        return jellyfish.levenshtein_distance(s1, s2)

    def _calculate_lcs(self, s1: str, s2: str) -> float:
        if len(s1) == 0 or len(s2) == 0:
            return 0
        lcs_length = pylcs.lcs_sequence_length(s1, s2)
        return lcs_length / min(len(s1), len(s2))

    def _calculate_n_diff(self, s1: str, s2: str) -> float:
        if len(s1) == 0:
            return 0
        diff = difflib.ndiff(s1, s2)
        diff_count = 0
        for line in diff:
            if line.startswith("-"):
                diff_count += 1
        return 1 - (diff_count / len(s1))

    def _calculate_n_gram(self, s1: str, s2: str) -> float:
        return strsimpy.ngram.NGram(3).distance(s1, s2)

    def _calculate_q_gram(self, s1: str, s2: str) -> int:
        return strsimpy.qgram.QGram(3).distance(s1, s2)

    def _calculate_jaccard(self, s1: str, s2: str) -> float:
        intersection = len(set(s1).intersection(set(s2)))
        union = len(set(s1).union(set(s2)))
        return intersection / union

    def _calculate_cosine(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0
        cosine = strsimpy.cosine.Cosine(3)
        p1 = cosine.get_profile(s1)
        p2 = cosine.get_profile(s2)
        if cosine._norm(p1) == 0 or cosine._norm(p2) == 0:
            return 0
        return cosine.similarity_profiles(p1, p2)

    def _calculate_fuzzy_ratio(self, s1: str, s2: str) -> float:
        return fuzz.partial_ratio(s1, s2)

    def _calculate_jaro(self, s1: str, s2: str) -> float:
        return jellyfish.jaro_similarity(s1, s2)

    def _calculate_hamming(self, s1: str, s2: str) -> float:
        return jellyfish.hamming_distance(s1, s2)

    def _calculate_jaro_winkler(self, s1: str, s2: str) -> float:
        return jellyfish.jaro_winkler_similarity(s1, s2)

    def _calculate_damerau_levenshtein(self, s1: str, s2: str) -> float:
        if len(s1) == 0:
            return 0
        return jellyfish.damerau_levenshtein_distance(s1, s2) / len(s1)


class BaseTextSimilarityMapper:
    def __init__(self) -> None:
        self._mapping: dict[str, Callable[[str, str], float]] = {}

    def register_method(self, name: str, method: Callable[[str, str], float]) -> None:
        self._mapping[name] = method

    def get_method(self, name: str):
        method = self._mapping.get(name)
        if method is None:
            raise ValueError(f"Method '{name}' is not registered.")
        return method


class TextSimilarityMapper(BaseExtractorAlgorithm, BaseTextSimilarityMapper):
    def __init__(self):
        super().__init__()

        # Text Similarity Algorithms
        self.register_method("levenshtein", self._calculate_levenshtein)
        self.register_method("lcs", self._calculate_lcs)
        self.register_method("n_diff", self._calculate_n_diff)
        self.register_method("n_gram", self._calculate_n_gram)
        self.register_method("q_gram", self._calculate_q_gram)
        self.register_method("jaccard", self._calculate_jaccard)
        self.register_method("cosine", self._calculate_cosine)
        self.register_method("fuzzy", self._calculate_fuzzy_ratio)
        self.register_method("jaro", self._calculate_jaro)
        self.register_method("hamming", self._calculate_hamming)
        self.register_method("jaro_winkler", self._calculate_jaro_winkler)
        self.register_method("damerau_levenshtein", self._calculate_damerau_levenshtein)
