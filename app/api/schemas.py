from typing import Any, List, Optional
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    document: str = Field(description="Name or path of the source document")
    chunk_id: str = Field(description="Unique ID of the cited chunk")
    page: Optional[int] = Field(default=None, description="Page number for PDFs (1-indexed), or null")
    excerpt: str = Field(description="Verbatim excerpt from the source chunk")
    score: float = Field(description="Similarity score between question and chunk")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="The user query to answer")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override the number of chunks to retrieve")


class AskResponse(BaseModel):
    answer: str = Field(description="Synthesized answer strictly grounded in the retrieved context")
    answerable: bool = Field(description="True if the question could be answered from provided documents")
    sources: List[SourceItem] = Field(default_factory=list, description="List of verified source excerpts")


class DocumentInfo(BaseModel):
    id: str = Field(description="Unique document ID (derived from content hash)")
    document_id: str = Field(default="", description="Unique document ID (alias for id)")
    filename: str = Field(description="Original filename")
    content_hash: str = Field(description="SHA-256 hash of file content")
    chunk_count: int = Field(description="Number of chunks indexed")
    page_count: Optional[int] = Field(default=None, description="Total pages if PDF")
    file_size_bytes: int = Field(description="File size in bytes")
    created_at: str = Field(description="ISO timestamp of ingestion")

    def model_post_init(self, __context: Any) -> None:
        if not self.document_id and self.id:
            self.document_id = self.id
        elif not self.id and self.document_id:
            self.id = self.document_id


class FileUploadResult(BaseModel):
    id: Optional[str] = Field(default=None, description="Document ID")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    filename: str = Field(description="Original filename")
    status: str = Field(description="Upload status: 'indexed', 'duplicate', or 'failed'")
    total_chunks: int = Field(default=0, description="Total chunks indexed")
    pages: Optional[int] = Field(default=None, description="Page count if PDF, or null")
    preview: Optional[str] = Field(default=None, description="First 300 characters of extracted text")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def model_post_init(self, __context: Any) -> None:
        if self.document_id and not self.id:
            self.id = self.document_id
        elif self.id and not self.document_id:
            self.document_id = self.id


class UploadResponse(BaseModel):
    files: List[FileUploadResult] = Field(default_factory=list, description="Per-file upload result status")
    uploaded: List[FileUploadResult] = Field(default_factory=list, description="List of indexed files")
    skipped: List[str] = Field(default_factory=list, description="List of duplicate filenames")
    message: str = Field(description="Status summary message")


class ChunkDetail(BaseModel):
    chunk_id: str = Field(description="Unique chunk identifier")
    index: int = Field(description="0-based index of chunk within the document")
    page: Optional[int] = Field(default=None, description="1-based page number if PDF, or null")
    text: str = Field(description="Chunk content text")
    char_count: int = Field(description="Character count of chunk text")


class DocumentChunksResponse(BaseModel):
    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original document filename")
    total_chunks: int = Field(description="Total chunks in document")
    total_pages: Optional[int] = Field(default=None, description="Total pages if PDF, or null")
    chunk_size: int = Field(description="Target chunk size configured in system")
    chunk_overlap: int = Field(description="Chunk overlap configured in system")
    chunks: List[ChunkDetail] = Field(default_factory=list, description="List of document chunks sorted by index")


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo] = Field(description="List of ingested documents")
    total: int = Field(description="Total document count")


class DeleteResponse(BaseModel):
    message: str = Field(description="Status confirmation")
    document_id: str = Field(description="ID of deleted document")


class HealthResponse(BaseModel):
    status: str = Field(description="Overall service health status ('ok' or 'degraded')")
    index_loaded: bool = Field(description="True if vector store index is active and ready")
    active_sessions: int = Field(default=0, description="Total active sessions")
    document_count: Optional[int] = Field(default=None, description="Optional document count")
    chunk_count: Optional[int] = Field(default=None, description="Optional chunk count")
    embedding_model: str = Field(description="Active embedding model name")
    llm_provider: str = Field(description="Configured LLM provider")


class SessionDeleteResponse(BaseModel):
    session_id: str = Field(description="The session identifier")
    deleted: bool = Field(description="True if session data was found and deleted, False otherwise")
    message: str = Field(description="Status confirmation message")


class ErrorDetail(BaseModel):
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    details: Optional[Any] = Field(default=None, description="Optional diagnostic details")


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(description="Error payload")
