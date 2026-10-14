from typing import List, Optional
from app.core.embeddings import BaseEmbeddingClient
from app.core.vector_store import BaseVectorStore, SearchResult
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.retriever")


class Retriever:
    """Retrieves relevant, diverse text chunks grounded above a similarity threshold."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_client: BaseEmbeddingClient,
        similarity_threshold: float = 0.55,
        default_top_k: int = 4,
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        threshold_override: Optional[float] = None,
    ) -> List[SearchResult]:
        k = top_k if top_k is not None and top_k > 0 else self.default_top_k
        threshold = threshold_override if threshold_override is not None else self.similarity_threshold

        query_cleaned = query.strip()
        if not query_cleaned:
            return []

        # Embed query
        query_vector = self.embedding_client.embed_text(query_cleaned)

        # Over-fetch candidates to allow diversity filtering
        candidate_k = max(k * 2, k + 4)
        candidates = self.vector_store.search(
            session_id=session_id,
            query_embedding=query_vector,
            top_k=candidate_k,
        )

        if not candidates:
            logger.info("Retriever found 0 candidates in vector store")
            return []

        # Filter by similarity threshold
        filtered = [c for c in candidates if c.score >= threshold]
        logger.info(
            f"Retrieved {len(candidates)} candidates, {len(filtered)} met similarity threshold ({threshold})"
        )

        if not filtered:
            return []

        # Diversify: prevent a single chunk from crowding out other relevant sources
        diversified = self._diversify_results(filtered, max_results=k)
        return diversified

    def _diversify_results(
        self, results: List[SearchResult], max_results: int
    ) -> List[SearchResult]:
        """Preserve top results while ensuring multi-document / multi-section variety."""
        selected: List[SearchResult] = []
        doc_counts: dict[str, int] = {}
        seen_texts: set[str] = set()

        # First pass: pick best scoring chunks ensuring no document dominates exclusively
        max_per_doc = max(2, max_results // 2)

        for item in results:
            text_signature = item.text[:120].strip()
            if text_signature in seen_texts:
                continue

            count = doc_counts.get(item.document_id, 0)
            if count < max_per_doc:
                selected.append(item)
                seen_texts.add(text_signature)
                doc_counts[item.document_id] = count + 1

            if len(selected) >= max_results:
                break

        # If we still have room, fill with remaining candidates
        if len(selected) < max_results:
            for item in results:
                text_signature = item.text[:120].strip()
                if item not in selected and text_signature not in seen_texts:
                    selected.append(item)
                    seen_texts.add(text_signature)
                if len(selected) >= max_results:
                    break

        return selected
