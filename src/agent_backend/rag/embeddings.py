"""Text embedders.

`HashingEmbedder` is a deterministic, dependency-free feature-hashing embedder (unigrams +
bigrams, signed hashing, L2-normalised). It captures lexical similarity only, but it lets
the whole RAG pipeline run offline and in tests. `OpenAIEmbedder` is the drop-in semantic
option for production.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from agent_backend.config import Settings

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a an and are as at be by for from how i in is it of on or that the this to was what "
        "when where which who why with you your"
    ).split()
)


class Embedder(ABC):
    dim: int

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
        return words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * (2.0 if "_" in feature else 1.0)
        return _normalise(vector)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str | None, model: str, dim: int, client: Any | None = None):
        if client is None:
            import openai

            client = openai.AsyncOpenAI(api_key=api_key)
        self._client = client
        self.model = model
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dim
        )
        return [item.embedding for item in response.data]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    return HashingEmbedder(dim=settings.embedding_dim)
