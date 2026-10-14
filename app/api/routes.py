import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Header, Query, Request, UploadFile, status
from app.api.errors import (
    DocumentNotFoundError,
    InvalidSessionIdError,
    SessionLimitExceededError,
    VectorStoreNotReadyError,
)
from app.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkDetail,
    DeleteResponse,
    DocumentChunksResponse,
    DocumentInfo,
    DocumentListResponse,
    FileUploadResult,
    HealthResponse,
    SessionDeleteResponse,
    UploadResponse,
)
from app.core.ingestion import IngestionResult
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.routes")
router = APIRouter()


def get_session_id(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> str:
    """Dependency that reads and validates X-Session-ID UUID header."""
    if not x_session_id or not x_session_id.strip():
        raise InvalidSessionIdError("Header 'X-Session-ID' is required")

    try:
        parsed_uuid = uuid.UUID(x_session_id.strip())
        valid_id = str(parsed_uuid)
    except (ValueError, AttributeError, TypeError):
        raise InvalidSessionIdError(f"Header 'X-Session-ID' must be a valid UUID, got '{x_session_id}'")

    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store:
        vector_store.touch_session(valid_id)

    return valid_id


def _handle_delete_session(request: Request, raw_session_id: Optional[str]) -> SessionDeleteResponse:
    if not raw_session_id or not raw_session_id.strip():
        raise InvalidSessionIdError("Session ID is required for cleanup")

    try:
        parsed = str(uuid.UUID(raw_session_id.strip()))
    except (ValueError, AttributeError, TypeError):
        raise InvalidSessionIdError(f"Session ID must be a valid UUID, got '{raw_session_id}'")

    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise VectorStoreNotReadyError()

    deleted = vector_store.delete_session(parsed)
    msg = f"Session '{parsed}' deleted successfully" if deleted else f"Session '{parsed}' was not found or already deleted"
    return SessionDeleteResponse(session_id=parsed, deleted=deleted, message=msg)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Public health check endpoint indicating vector store readiness and active session count."""
    try:
        vector_store = getattr(request.app.state, "vector_store", None)
        settings = request.app.state.settings

        if vector_store is None:
            return HealthResponse(
                status="degraded",
                index_loaded=False,
                active_sessions=0,
                embedding_model=settings.embedding_model,
                llm_provider=settings.llm_provider,
            )

        active_sessions = vector_store.get_active_session_count()
        return HealthResponse(
            status="ok",
            index_loaded=True,
            active_sessions=active_sessions,
            embedding_model=settings.embedding_model,
            llm_provider=settings.llm_provider,
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            index_loaded=False,
            active_sessions=0,
            embedding_model="unknown",
            llm_provider="unknown",
        )


@router.delete("/session", response_model=SessionDeleteResponse)
async def delete_session(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Deletes a session collection and all associated document records."""
    return _handle_delete_session(request, x_session_id)


@router.post("/session/end", response_model=SessionDeleteResponse)
async def end_session(
    request: Request,
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """POST-compatible session termination endpoint for navigator.sendBeacon."""
    target_id = session_id or x_session_id
    return _handle_delete_session(request, target_id)


@router.post("/documents", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more .pdf, .txt, or .md files"),
    session_id: str = Depends(get_session_id),
):
    """Session-isolated multipart upload for indexing documents."""
    settings = request.app.state.settings
    ingestion_service = request.app.state.ingestion_service
    embedding_client = request.app.state.embedding_client
    vector_store = request.app.state.vector_store

    # 1. Enforce session document count limit
    current_docs = vector_store.list_documents(session_id)
    if len(current_docs) + len(files) > settings.max_files_per_session:
        raise SessionLimitExceededError(
            f"Session document limit exceeded: maximum allowed is {settings.max_files_per_session} files per session (current: {len(current_docs)}, incoming: {len(files)})"
        )

    # 2. Pre-read payloads and enforce session total upload size limit
    file_payloads: List[tuple[str, bytes]] = []
    total_incoming_bytes = 0
    for file in files:
        fname = file.filename or "unknown"
        content = await file.read()
        file_payloads.append((fname, content))
        total_incoming_bytes += len(content)

    current_bytes = vector_store.get_total_upload_bytes(session_id)
    max_bytes = settings.max_total_upload_mb * 1024 * 1024
    if current_bytes + total_incoming_bytes > max_bytes:
        raise SessionLimitExceededError(
            f"Session total upload size limit exceeded: maximum allowed is {settings.max_total_upload_mb} MB per session"
        )

    uploaded_results: List[FileUploadResult] = []
    file_results: List[FileUploadResult] = []
    skipped_files: List[str] = []

    for filename, content in file_payloads:
        try:
            # Ingest and validate file
            ingest_result: IngestionResult = ingestion_service.process_document(
                filename=filename,
                content=content,
            )

            # Check for duplicate document content hash within this session
            if vector_store.has_document_by_hash(session_id, ingest_result.content_hash):
                logger.info(f"Skipping duplicate file in session {session_id}: {filename}")
                preview = (
                    ingest_result.extracted_text[:300]
                    if getattr(ingest_result, "extracted_text", None)
                    else ("\n\n".join(c.text for c in ingest_result.chunks))[:300]
                )
                dup_res = FileUploadResult(
                    id=ingest_result.document_id,
                    document_id=ingest_result.document_id,
                    filename=filename,
                    status="duplicate",
                    total_chunks=len(ingest_result.chunks),
                    pages=ingest_result.page_count,
                    preview=preview,
                    error=None,
                )
                file_results.append(dup_res)
                skipped_files.append(filename)
                continue

            # Embed all extracted chunks
            chunk_texts = [c.text for c in ingest_result.chunks]
            embeddings = embedding_client.embed_batch(chunk_texts)

            # Build DocumentInfo record
            doc_info = DocumentInfo(
                id=ingest_result.document_id,
                document_id=ingest_result.document_id,
                filename=ingest_result.filename,
                content_hash=ingest_result.content_hash,
                chunk_count=len(ingest_result.chunks),
                page_count=ingest_result.page_count,
                file_size_bytes=ingest_result.file_size_bytes,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Commit chunks to session-scoped vector store
            vector_store.add_chunks(
                session_id=session_id,
                doc_info=doc_info,
                chunks=ingest_result.chunks,
                embeddings=embeddings,
            )

            preview = (
                ingest_result.extracted_text[:300]
                if getattr(ingest_result, "extracted_text", None)
                else ("\n\n".join(c.text for c in ingest_result.chunks))[:300]
            )
            success_res = FileUploadResult(
                id=ingest_result.document_id,
                document_id=ingest_result.document_id,
                filename=filename,
                status="indexed",
                total_chunks=len(ingest_result.chunks),
                pages=ingest_result.page_count,
                preview=preview,
                error=None,
            )
            uploaded_results.append(success_res)
            file_results.append(success_res)

        except Exception as e:
            logger.error(f"Failed to process file '{filename}' in session {session_id}: {e}")
            err_msg = getattr(e, "message", str(e))
            failed_res = FileUploadResult(
                id=None,
                document_id=None,
                filename=filename,
                status="failed",
                total_chunks=0,
                pages=None,
                preview=None,
                error=err_msg,
            )
            file_results.append(failed_res)

    indexed_count = len(uploaded_results)
    skipped_count = len(skipped_files)
    failed_count = sum(1 for f in file_results if f.status == "failed")
    summary_msg = f"Processed {len(files)} files: {indexed_count} indexed, {skipped_count} skipped, {failed_count} failed"

    return UploadResponse(
        files=file_results,
        uploaded=uploaded_results,
        skipped=skipped_files,
        message=summary_msg,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    session_id: str = Depends(get_session_id),
):
    """Retrieve all ingested documents in the current session."""
    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise VectorStoreNotReadyError()

    docs = vector_store.list_documents(session_id)
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    request: Request,
    document_id: str,
    session_id: str = Depends(get_session_id),
):
    """Retrieve all chunks belonging to a document in the current session, sorted by index."""
    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise VectorStoreNotReadyError()

    doc_info = vector_store.get_document(session_id, document_id)
    if not doc_info:
        raise DocumentNotFoundError(document_id)

    settings = request.app.state.settings
    raw_chunks = vector_store.get_document_chunks(session_id, document_id)
    chunks = [
        ChunkDetail(
            chunk_id=c["chunk_id"],
            index=c["index"],
            page=c.get("page"),
            text=c["text"],
            char_count=c.get("char_count", len(c["text"])),
        )
        for c in raw_chunks
    ]

    return DocumentChunksResponse(
        document_id=doc_info.id,
        filename=doc_info.filename,
        total_chunks=len(chunks),
        total_pages=doc_info.page_count,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunks=chunks,
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    request: Request,
    document_id: str,
    session_id: str = Depends(get_session_id),
):
    """Delete a document and purge all associated chunks from the session collection."""
    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise VectorStoreNotReadyError()

    deleted = vector_store.delete_document(session_id, document_id)
    if not deleted:
        raise DocumentNotFoundError(document_id)

    return DeleteResponse(
        message=f"Document '{document_id}' successfully deleted",
        document_id=document_id,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: Request,
    payload: AskRequest,
    session_id: str = Depends(get_session_id),
):
    """Answer questions using only retrieved and verified document context in the current session."""
    qa_service = getattr(request.app.state, "qa_service", None)
    if not qa_service:
        raise VectorStoreNotReadyError("QA Service is not initialized")

    response = qa_service.answer_question(
        question=payload.question,
        session_id=session_id,
        top_k=payload.top_k,
    )
    return response
