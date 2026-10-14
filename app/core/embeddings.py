import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional
import httpx
from app.api.errors import AppException
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.embeddings")


class BaseEmbeddingClient(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        pass


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    """Local embedding client using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self._model_name = model_name
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI API-based embedding client."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._dimension = 1536

    def embed_text(self, text: str) -> List[float]:
        batch = self.embed_batch([text])
        return batch[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": texts, "model": self._model}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                data_items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in data_items]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise AppException(
                status_code=503,
                code="EMBEDDING_API_ERROR",
                message=f"Failed to fetch embeddings from OpenAI: {str(e)}",
            )

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbeddingClient(BaseEmbeddingClient):
    """Deterministic, lightweight embedding client for testing without heavy models or network."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def _hash_embed(self, text: str) -> List[float]:
        words = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._dimension

        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            vec[idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_text(self, text: str) -> List[float]:
        return self._hash_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding_client(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
) -> BaseEmbeddingClient:
    provider = provider.lower().strip()
    if provider == "mock":
        return MockEmbeddingClient()
    elif provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided for OpenAI embeddings")
        return OpenAIEmbeddingClient(api_key=api_key, model=model_name)
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddingClient(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
