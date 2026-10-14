import uuid
from app.api.schemas import DocumentInfo
from app.core.chunking import Chunk
from app.core.embeddings import MockEmbeddingClient
from app.core.retriever import Retriever
from app.core.vector_store import ChromaVectorStore


def test_retriever_empty_store(temp_chroma_dir):
    store = ChromaVectorStore(persist_dir=temp_chroma_dir)
    embedder = MockEmbeddingClient()
    retriever = Retriever(vector_store=store, embedding_client=embedder, similarity_threshold=0.5)

    session_id = str(uuid.uuid4())
    results = retriever.retrieve("Any question", session_id=session_id)
    assert results == []


def test_retriever_ranking_and_threshold(temp_chroma_dir):
    store = ChromaVectorStore(persist_dir=temp_chroma_dir)
    embedder = MockEmbeddingClient()
    retriever = Retriever(vector_store=store, embedding_client=embedder, similarity_threshold=0.4)
    session_id = str(uuid.uuid4())

    # Document 1: Storage Pricing
    doc1 = DocumentInfo(
        id="doc_pricing",
        filename="pricing.md",
        content_hash="hash1",
        chunk_count=2,
        page_count=None,
        file_size_bytes=500,
        created_at="2026-10-14T00:00:00Z",
    )
    chunks1 = [
        Chunk(
            chunk_id="doc_pricing_c0",
            document_id="doc_pricing",
            document_name="pricing.md",
            text="Standard storage tier pricing is twenty dollars per terabyte monthly.",
            chunk_index=0,
            char_start=0,
            char_end=70,
            page=None,
        ),
        Chunk(
            chunk_id="doc_pricing_c1",
            document_id="doc_pricing",
            document_name="pricing.md",
            text="Archive cold tier pricing is one dollar per terabyte monthly.",
            chunk_index=1,
            char_start=71,
            char_end=130,
            page=None,
        ),
    ]
    embeddings1 = embedder.embed_batch([c.text for c in chunks1])
    store.add_chunks(session_id, doc1, chunks1, embeddings1)

    # Document 2: Network Firewall
    doc2 = DocumentInfo(
        id="doc_firewall",
        filename="firewall.txt",
        content_hash="hash2",
        chunk_count=1,
        page_count=None,
        file_size_bytes=300,
        created_at="2026-10-14T00:00:00Z",
    )
    chunks2 = [
        Chunk(
            chunk_id="doc_firewall_c0",
            document_id="doc_firewall",
            document_name="firewall.txt",
            text="Default firewall rules block all incoming traffic on port 22 and port 80.",
            chunk_index=0,
            char_start=0,
            char_end=76,
            page=None,
        ),
    ]
    embeddings2 = embedder.embed_batch([c.text for c in chunks2])
    store.add_chunks(session_id, doc2, chunks2, embeddings2)

    # Search for storage pricing
    results = retriever.retrieve("storage tier pricing per terabyte monthly", session_id=session_id, top_k=2)
    assert len(results) >= 1
    assert "pricing" in results[0].text.lower()
    assert results[0].score >= 0.4

    # Search for completely unrelated terms with strict threshold
    unrelated = retriever.retrieve("quantum physics string theory graviton", session_id=session_id, threshold_override=0.85)
    assert len(unrelated) == 0


def test_retriever_document_deletion(temp_chroma_dir):
    store = ChromaVectorStore(persist_dir=temp_chroma_dir)
    embedder = MockEmbeddingClient()
    retriever = Retriever(vector_store=store, embedding_client=embedder, similarity_threshold=0.2)
    session_id = str(uuid.uuid4())

    doc = DocumentInfo(
        id="doc_to_delete",
        filename="temporary.md",
        content_hash="temp_hash",
        chunk_count=1,
        page_count=None,
        file_size_bytes=200,
        created_at="2026-10-14T00:00:00Z",
    )
    chunks = [
        Chunk(
            chunk_id="doc_to_delete_c0",
            document_id="doc_to_delete",
            document_name="temporary.md",
            text="Temporary confidential credentials for database cluster access.",
            chunk_index=0,
            char_start=0,
            char_end=60,
            page=None,
        )
    ]
    store.add_chunks(session_id, doc, chunks, embedder.embed_batch([chunks[0].text]))

    assert store.count(session_id) == 1
    assert store.get_document(session_id, "doc_to_delete") is not None

    # Delete
    deleted = store.delete_document(session_id, "doc_to_delete")
    assert deleted is True
    assert store.count(session_id) == 0
    assert store.get_document(session_id, "doc_to_delete") is None

    # Deleting non-existent returns False
    assert store.delete_document(session_id, "non_existent") is False

