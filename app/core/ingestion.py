import hashlib
import io
from dataclasses import dataclass
from typing import List, Optional
from pypdf import PdfReader
from app.api.errors import (
    DocumentEmptyError,
    FileSizeLimitError,
    UnsupportedMediaTypeError,
)
from app.core.chunking import Chunk, TextChunker
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.ingestion")


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    content_hash: str
    file_size_bytes: int
    chunks: List[Chunk]
    page_count: Optional[int] = None
    extracted_text: str = ""


class DocumentIngestionService:
    def __init__(
        self,
        chunker: TextChunker,
        max_file_size_mb: int = 10,
        allowed_extensions: Optional[List[str]] = None,
    ):
        self.chunker = chunker
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_file_size_mb = max_file_size_mb
        self.allowed_extensions = [
            ext.lower() for ext in (allowed_extensions or [".pdf", ".txt", ".md"])
        ]

    def validate_file(self, filename: str, content: bytes) -> None:
        """Validate filename extension, size, and non-empty byte payload."""
        if not filename or "." not in filename:
            raise UnsupportedMediaTypeError("unknown", self.allowed_extensions)

        extension = "." + filename.rsplit(".", 1)[-1].lower()
        if extension not in self.allowed_extensions:
            raise UnsupportedMediaTypeError(extension, self.allowed_extensions)

        if len(content) == 0:
            raise DocumentEmptyError(f"File '{filename}' is empty (0 bytes)")

        if len(content) > self.max_file_size_bytes:
            raise FileSizeLimitError(
                self.max_file_size_mb,
                details={"filename": filename, "size_bytes": len(content)},
            )

    @staticmethod
    def calculate_hash(content: bytes) -> str:
        """Calculate SHA-256 hash of file content for idempotency."""
        return hashlib.sha256(content).hexdigest()

    def process_document(self, filename: str, content: bytes) -> IngestionResult:
        """Validate, extract text from supported format, and produce structured chunks."""
        self.validate_file(filename, content)
        content_hash = self.calculate_hash(content)
        doc_id = content_hash[:16]
        extension = "." + filename.rsplit(".", 1)[-1].lower()

        if extension == ".pdf":
            chunks, page_count, extracted_text = self._extract_pdf(content, doc_id, filename)
        elif extension in [".txt", ".md"]:
            chunks, page_count, extracted_text = self._extract_text(content, doc_id, filename)
        else:
            raise UnsupportedMediaTypeError(extension, self.allowed_extensions)

        if not chunks:
            raise DocumentEmptyError(
                f"File '{filename}' yielded no extractable text. Scanned PDFs without OCR or blank files are not supported."
            )

        logger.info(
            f"Ingested '{filename}' (doc_id={doc_id}) into {len(chunks)} chunks across {page_count or 1} pages"
        )
        return IngestionResult(
            document_id=doc_id,
            filename=filename,
            content_hash=content_hash,
            file_size_bytes=len(content),
            chunks=chunks,
            page_count=page_count,
            extracted_text=extracted_text,
        )

    def _extract_text(
        self, content: bytes, doc_id: str, filename: str
    ) -> tuple[List[Chunk], None, str]:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as e:
                raise DocumentEmptyError(f"Could not decode text in '{filename}': {str(e)}")

        stripped = text.strip()
        if not stripped:
            raise DocumentEmptyError(f"Text file '{filename}' contains only whitespace")

        chunks = self.chunker.chunk_text(
            text=stripped,
            document_id=doc_id,
            document_name=filename,
            page=None,
        )
        return chunks, None, stripped

    def _extract_pdf(
        self, content: bytes, doc_id: str, filename: str
    ) -> tuple[List[Chunk], int, str]:
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as e:
            raise DocumentEmptyError(f"Failed to parse PDF '{filename}': {str(e)}")

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise DocumentEmptyError(f"PDF '{filename}' has 0 pages")

        all_chunks: List[Chunk] = []
        page_texts: List[str] = []
        total_extracted_chars = 0
        chunk_offset = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_idx} of {filename}: {e}")
                page_text = ""

            page_stripped = page_text.strip()
            total_extracted_chars += len(page_stripped)

            if page_stripped:
                page_texts.append(page_stripped)
                page_chunks = self.chunker.chunk_text(
                    text=page_stripped,
                    document_id=doc_id,
                    document_name=filename,
                    page=page_idx,
                    start_index=chunk_offset,
                )
                all_chunks.extend(page_chunks)
                chunk_offset += len(page_chunks)

        if total_extracted_chars == 0:
            raise DocumentEmptyError(
                f"PDF '{filename}' contains no extractable text. It may be a scanned document or image-only PDF."
            )

        full_pdf_text = "\n\n".join(page_texts)
        return all_chunks, num_pages, full_pdf_text
