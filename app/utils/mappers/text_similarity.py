import difflib
from typing import Callable

import jellyfish
import numpy as np


class BaseExtractorAlgorithm:
    def _calculate_levenshtein(self, s1: str, s2: str) -> int:
        if len(s1) == 0:
            return 0
        return jellyfish.levenshtein_distance(s1, s2) / len(s1)

    def _calculate_jaro(self, s1: str, s2: str) -> float:
        return jellyfish.jaro_similarity(s1, s2)

    def _calculate_lcs(self, s1: str, s2: str) -> float:
        m, n = len(s1), len(s2)
        if m == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_length = dp[m][n]
        return lcs_length / m

    def _calculate_n_diff(self, s1: str, s2: str) -> float:
        if len(s1) == 0:
            return 0

        diff = difflib.ndiff(s1, s2)
        diff_count = 0

        for line in diff:
            if line.startswith("-"):
                diff_count += 1
        return 1 - (diff_count / len(s1))

    def _calculate_damerau_levenshtein(self, s1: str, s2: str) -> float:
        if len(s1) == 0:
            return 0
        return jellyfish.damerau_levenshtein_distance(s1, s2) / len(s1)

    def _calculate_jaccard(self, s1: str, s2: str) -> float:
        return jellyfish.jaccard_similarity(s1, s2, ngram_size=1)

    def _calculate_hamming(self, s1: str, s2: str) -> float:
        if len(s1) == 0:
            return 0
        return jellyfish.hamming_distance(s1, s2) / len(s1)

    def _calculate_jaro_winkler(self, s1: str, s2: str) -> float:
        return jellyfish.jaro_winkler_similarity(s1, s2, long_tolerance=True)


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
        self.register_method("jaro", self._calculate_jaro)
        self.register_method("n_diff", self._calculate_n_diff)
        self.register_method("damerau_levenshtein", self._calculate_damerau_levenshtein)
        self.register_method("jaccard", self._calculate_jaccard)
        self.register_method("hamming", self._calculate_hamming)
        self.register_method("jaro_winkler", self._calculate_jaro_winkler)
