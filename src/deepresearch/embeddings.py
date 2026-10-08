from __future__ import annotations

import math
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    """Minimal embedding boundary used by local passage retrieval."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingProvider:
    """Call a Qwen or other OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimensions: int = 512,
        batch_size: int = 10,
        timeout: float = 45.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("An embedding base URL is required")
        if not model.strip():
            raise ValueError("An embedding model is required")
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        normalized_url = base_url.rstrip("/")
        self.endpoint = (
            normalized_url
            if normalized_url.endswith("/embeddings")
            else f"{normalized_url}/embeddings"
        )
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = httpx.post(
                self.endpoint,
                headers=headers,
                json={
                    "model": self.model,
                    "input": batch,
                    "dimensions": self.dimensions,
                    "encoding_format": "float",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("Embedding response is missing a data list")
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            if len(ordered) != len(batch):
                raise ValueError(
                    "Embedding response count does not match the request batch"
                )
            for item in ordered:
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise ValueError("Embedding response contains an invalid vector")
                parsed = [float(value) for value in vector]
                if len(parsed) != self.dimensions:
                    raise ValueError(
                        "Embedding response dimension does not match configuration"
                    )
                if not all(math.isfinite(value) for value in parsed):
                    raise ValueError("Embedding response contains a non-finite value")
                embeddings.append(parsed)
        return embeddings
