import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.errors import register_exception_handlers
from app.api.routes import router as api_router
from app.config import Settings, get_settings
from app.core.chunking import TextChunker
from app.core.embeddings import create_embedding_client
from app.core.ingestion import DocumentIngestionService
from app.core.llm import create_llm_client
from app.core.qa_service import QAService
from app.core.retriever import Retriever
from app.core.vector_store import ChromaVectorStore
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.main")


def initialize_services(app: FastAPI, settings: Settings) -> None:
    """Initialize application core components and attach to app state."""
    app.state.settings = settings

    logger.info(f"Initializing Vector Store with dir: {settings.chroma_persist_dir}")
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
    )
    app.state.vector_store = vector_store

    logger.info(f"Initializing Embeddings with provider: {settings.embedding_provider}")
    embedding_client = create_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.llm_api_key,
    )
    app.state.embedding_client = embedding_client

    logger.info("Initializing Text Chunker and Ingestion Service")
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    ingestion_service = DocumentIngestionService(
        chunker=chunker,
        max_file_size_mb=settings.max_file_size_mb,
        allowed_extensions=settings.allowed_extensions,
    )
    app.state.ingestion_service = ingestion_service

    logger.info("Initializing Retriever and LLM Client")
    retriever = Retriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        similarity_threshold=settings.similarity_threshold,
        default_top_k=settings.top_k,
    )
    app.state.retriever = retriever

    llm_client = create_llm_client(
        provider=settings.llm_provider,
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    app.state.llm_client = llm_client

    qa_service = QAService(
        retriever=retriever,
        llm_client=llm_client,
        default_top_k=settings.top_k,
    )
    app.state.qa_service = qa_service
    logger.info("All RAG services successfully initialized")


async def session_cleanup_worker(app: FastAPI, ttl_minutes: int):
    """Background task running every 60 seconds to prune idle sessions."""
    import asyncio
    while True:
        try:
            await asyncio.sleep(60)
            vector_store = getattr(app.state, "vector_store", None)
            if vector_store:
                deleted = vector_store.cleanup_idle_sessions(ttl_minutes)
                if deleted > 0:
                    logger.info(f"Background worker pruned {deleted} idle sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in session cleanup worker: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    settings = get_settings()
    initialize_services(app, settings)

    # Clean up leftover session collections from previous runs on startup
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store:
        vector_store.cleanup_leftover_collections()

    # Start periodic idle session TTL cleanup worker
    cleanup_task = asyncio.create_task(session_cleanup_worker(app, settings.session_ttl_minutes))

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Application shutdown completed")


def create_app(settings: Settings = None) -> FastAPI:
    app = FastAPI(
        title="Reliable RAG Knowledge Assistant",
        description="Production-grade grounded document QA service with citations and refusal guarantees",
        version="1.0.0",
        lifespan=lifespan,
    )

    register_exception_handlers(app)
    app.include_router(api_router)

    # Static file directory for minimal developer UI
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            index_path = os.path.join(static_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"status": "ok", "message": "Reliable RAG Knowledge Assistant API"}

    return app


app = create_app()
