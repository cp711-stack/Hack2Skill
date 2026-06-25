from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocess import tokenize


@dataclass
class RetrievalBackend:
    name: str
    available: bool


class HybridRetriever:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        self.backend = self._load_backend()

    def rank(self, jd_text: str, profiles: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        if profiles.empty:
            return profiles.copy()

        ranked = profiles.copy()
        texts = ranked["rich_profile"].fillna("").tolist()
        ranked["semantic_score"] = self._semantic_scores(jd_text, texts)
        ranked["bm25_score"] = self._bm25_scores(jd_text, texts)
        ranked["retrieval_score"] = (
            0.75 * ranked["semantic_score"] + 0.25 * ranked["bm25_score"]
        )
        return (
            ranked.sort_values("retrieval_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    def _load_backend(self) -> RetrievalBackend:
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            return RetrievalBackend(name=f"sentence-transformers/{self.embedding_model_name}", available=True)
        except Exception:
            self._embedding_model = None
            return RetrievalBackend(name="tfidf-fallback", available=True)

    def _semantic_scores(self, jd_text: str, profile_texts: list[str]) -> np.ndarray:
        if self._embedding_model is not None:
            documents = [jd_text] + profile_texts
            embeddings = self._embedding_model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = _cosine_scores(embeddings[0], embeddings[1:])
            return _normalize(scores)

        scores = _tfidf_cosine_scores(jd_text, profile_texts)
        return _normalize(scores)

    def _bm25_scores(self, jd_text: str, profile_texts: list[str]) -> np.ndarray:
        tokenized_profiles = [tokenize(text) for text in profile_texts]
        query_tokens = tokenize(jd_text)
        scores = _simple_bm25_scores(tokenized_profiles, query_tokens)
        return _normalize(scores)


def _normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score == min_score:
        return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _cosine_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    denominator = query_norm * document_norms
    denominator = np.where(denominator == 0, 1, denominator)
    return np.dot(document_vectors, query_vector) / denominator


def _tfidf_cosine_scores(query_text: str, profile_texts: list[str]) -> np.ndarray:
    documents = [tokenize(query_text)] + [tokenize(text) for text in profile_texts]
    vocabulary = sorted({token for document in documents for token in document})
    if not vocabulary:
        return np.zeros(len(profile_texts), dtype=float)

    token_index = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)
    document_frequency = np.zeros(len(vocabulary), dtype=float)

    for row_index, document in enumerate(documents):
        seen_in_document: set[int] = set()
        for token in document:
            column_index = token_index[token]
            matrix[row_index, column_index] += 1
            seen_in_document.add(column_index)
        for column_index in seen_in_document:
            document_frequency[column_index] += 1

    idf = np.log((1 + len(documents)) / (1 + document_frequency)) + 1
    matrix *= idf
    return _cosine_scores(matrix[0], matrix[1:])


def _simple_bm25_scores(
    documents: list[list[str]],
    query_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    if not documents:
        return np.array([], dtype=float)

    doc_count = len(documents)
    doc_lengths = np.array([len(document) for document in documents], dtype=float)
    avg_doc_length = float(np.mean(doc_lengths)) if np.any(doc_lengths) else 1.0
    query_terms = set(query_tokens)

    document_frequency: dict[str, int] = {}
    for document in documents:
        for token in set(document):
            if token in query_terms:
                document_frequency[token] = document_frequency.get(token, 0) + 1

    scores = np.zeros(doc_count, dtype=float)
    for index, document in enumerate(documents):
        term_counts: dict[str, int] = {}
        for token in document:
            if token in query_terms:
                term_counts[token] = term_counts.get(token, 0) + 1

        for token, frequency in term_counts.items():
            df = document_frequency.get(token, 0)
            idf = np.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_lengths[index] / avg_doc_length)
            scores[index] += idf * (frequency * (k1 + 1)) / denominator

    return scores
