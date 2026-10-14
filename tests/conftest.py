import io
import shutil
import tempfile
from typing import Generator, List
import pytest
from pypdf import PdfWriter


@pytest.fixture
def temp_chroma_dir() -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_markdown() -> str:
    return """# Acme Cloud Storage Policy

## Overview
Acme Cloud Storage provides three primary tiers: Standard, Cool, and Archive.

### Standard Tier
Standard tier is designed for frequently accessed data with an SLA of 99.99% availability.
Pricing is $0.020 per GB per month. No retrieval fee is charged.

### Cool Tier
Cool tier is designed for data accessed less than once a month with 99.9% availability.
Pricing is $0.010 per GB per month with a minimum storage duration of 30 days.

## Data Retention and Security
All buckets must enable server-side encryption with customer-managed keys (CMEK) or AES-256.
Access logs must be retained for at least 365 days in compliant storage.
"""


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a minimal valid 2-page PDF in memory for testing."""
    writer = PdfWriter()
    # Add page with text annotations / contents
    # Since pypdf PdfWriter can write blank pages, let's create a real PDF with text stream
    # or write raw PDF stream
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R 4 0 R]/Count 2>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 5 0 R/Resources<</Font<</F1 7 0 R>>>>>>endobj\n"
        b"4 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 6 0 R/Resources<</Font<</F1 7 0 R>>>>>>endobj\n"
        b"5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 712 Td (Page 1: Engineering Onboarding) Tj ET\nendstream\nendobj\n"
        b"6 0 obj<</Length 42>>stream\nBT /F1 12 Tf 72 712 Td (Page 2: Security Best Practices) Tj ET\nendstream\nendobj\n"
        b"7 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000216 00000 n \n0000000317 00000 n \n0000000412 00000 n \n0000000505 00000 n \ntrailer<</Size 8/Root 1 0 R>>\nstartxref\n567\n%%EOF\n"
    )
    return pdf_content


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """A valid PDF structure with 1 page but no text stream (simulating scanned/empty page)."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000109 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n185\n%%EOF\n"
    )
    return pdf_content


@pytest.fixture
def test_app_and_state(temp_chroma_dir):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.core.chunking import TextChunker
    from app.core.embeddings import MockEmbeddingClient
    from app.core.ingestion import DocumentIngestionService
    from app.core.llm import MockLLMClient
    from app.core.qa_service import QAService
    from app.core.retriever import Retriever
    from app.core.vector_store import ChromaVectorStore
    from app.main import create_app

    settings = Settings(
        chroma_persist_dir=temp_chroma_dir,
        embedding_provider="mock",
        llm_provider="mock",
        similarity_threshold=0.3,
        chunk_size=300,
        chunk_overlap=50,
    )
    app = create_app(settings)

    vector_store = ChromaVectorStore(persist_dir=temp_chroma_dir)
    embedding_client = MockEmbeddingClient()
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    ingestion_service = DocumentIngestionService(chunker=chunker)
    retriever = Retriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        similarity_threshold=0.3,
        default_top_k=4,
    )
    llm_client = MockLLMClient()
    qa_service = QAService(retriever=retriever, llm_client=llm_client)

    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.embedding_client = embedding_client
    app.state.ingestion_service = ingestion_service
    app.state.retriever = retriever
    app.state.llm_client = llm_client
    app.state.qa_service = qa_service

    import uuid
    client = TestClient(app)
    default_session_id = str(uuid.uuid4())
    client.headers["X-Session-ID"] = default_session_id
    app.state.default_session_id = default_session_id
    return client, app.state


@pytest.fixture
def test_client(test_app_and_state):
    client, _ = test_app_and_state
    return client


@pytest.fixture
def session_id(test_app_and_state) -> str:
    _, state = test_app_and_state
    return state.default_session_id


@pytest.fixture
def session_headers(session_id: str) -> dict:
    return {"X-Session-ID": session_id}

