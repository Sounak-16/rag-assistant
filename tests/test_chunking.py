import pytest
from app.api.errors import (
    DocumentEmptyError,
    FileSizeLimitError,
    UnsupportedMediaTypeError,
)
from app.core.chunking import TextChunker
from app.core.ingestion import DocumentIngestionService


def test_chunker_initialization_validation():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=100, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_empty_string():
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(
        text="   \n\n   ",
        document_id="doc1",
        document_name="test.txt",
    )
    assert chunks == []


def test_chunker_preserves_headings_and_paragraphs(sample_markdown):
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_text(
        text=sample_markdown,
        document_id="doc123",
        document_name="policy.md",
    )
    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= 400
        assert chunk.document_id == "doc123"
        assert chunk.document_name == "policy.md"
        assert chunk.chunk_id.startswith("doc123_p0_c")
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start

    # Check that headings remain at start or boundary
    first_chunk = chunks[0]
    assert "Acme Cloud Storage Policy" in first_chunk.text


def test_chunker_overlap_consistency():
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta. " * 15
    )
    chunker = TextChunker(chunk_size=150, chunk_overlap=40)
    chunks = chunker.chunk_text(
        text=text,
        document_id="doc_overlap",
        document_name="overlap.txt",
    )
    assert len(chunks) >= 3

    # Check consecutive chunks share overlap content
    for i in range(len(chunks) - 1):
        c1 = chunks[i].text
        c2 = chunks[i+1].text
        # Tail of c1 or words from tail should appear in head of c2
        words_c1 = c1.split()
        assert any(word in c2 for word in words_c1[-3:])


def test_ingestion_text_and_markdown(sample_markdown):
    chunker = TextChunker(chunk_size=250, chunk_overlap=50)
    service = DocumentIngestionService(chunker=chunker, max_file_size_mb=2)

    # Markdown test
    md_bytes = sample_markdown.encode("utf-8")
    result_md = service.process_document("handbook.md", md_bytes)
    assert result_md.filename == "handbook.md"
    assert result_md.page_count is None
    assert len(result_md.chunks) >= 2
    assert result_md.content_hash == service.calculate_hash(md_bytes)

    # Plain text test
    txt_content = b"This is a simple plain text configuration file with key=value pairs."
    result_txt = service.process_document("config.txt", txt_content)
    assert result_txt.filename == "config.txt"
    assert len(result_txt.chunks) == 1
    assert result_txt.chunks[0].page is None


def test_ingestion_valid_pdf(sample_pdf_bytes):
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    service = DocumentIngestionService(chunker=chunker)

    result = service.process_document("guide.pdf", sample_pdf_bytes)
    assert result.filename == "guide.pdf"
    assert result.page_count == 2
    assert len(result.chunks) >= 2

    # Verify page metadata is captured
    pages = [c.page for c in result.chunks]
    assert 1 in pages
    assert 2 in pages


def test_ingestion_empty_file_rejected():
    service = DocumentIngestionService(chunker=TextChunker())
    with pytest.raises(DocumentEmptyError, match="is empty"):
        service.process_document("empty.txt", b"")


def test_ingestion_whitespace_only_rejected():
    service = DocumentIngestionService(chunker=TextChunker())
    with pytest.raises(DocumentEmptyError, match="whitespace"):
        service.process_document("empty.md", b"   \n\t  \n  ")


def test_ingestion_unsupported_media_type():
    service = DocumentIngestionService(chunker=TextChunker())
    with pytest.raises(UnsupportedMediaTypeError, match="Unsupported file extension '.docx'"):
        service.process_document("report.docx", b"dummy binary data")


def test_ingestion_size_limit_exceeded():
    service = DocumentIngestionService(chunker=TextChunker(), max_file_size_mb=1)
    huge_data = b"x" * (1024 * 1024 + 100)
    with pytest.raises(FileSizeLimitError):
        service.process_document("huge.txt", huge_data)


def test_ingestion_scanned_or_empty_pdf_rejected(empty_pdf_bytes):
    service = DocumentIngestionService(chunker=TextChunker())
    with pytest.raises(DocumentEmptyError, match="no extractable text"):
        service.process_document("scanned.pdf", empty_pdf_bytes)


def test_ingestion_content_hash_idempotency():
    service = DocumentIngestionService(chunker=TextChunker())
    data = b"Idempotent document content for hash verification"
    hash1 = service.calculate_hash(data)
    hash2 = service.calculate_hash(data)
    assert hash1 == hash2
    assert len(hash1) == 64
