import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.api.errors import VectorStoreNotReadyError
from app.api.schemas import DocumentInfo
from app.core.chunking import Chunk
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.vector_store")


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    page: Optional[int] = None
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0


@dataclass
class SessionState:
    session_id: str
    created_at: datetime
    last_active: datetime
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(
        self,
        session_id: str,
        doc_info: DocumentInfo,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def delete_document(self, session_id: str, document_id: str) -> bool:
        pass

    @abstractmethod
    def get_document(self, session_id: str, document_id: str) -> Optional[DocumentInfo]:
        pass

    @abstractmethod
    def has_document_by_hash(self, session_id: str, content_hash: str) -> bool:
        pass

    @abstractmethod
    def list_documents(self, session_id: str) -> List[DocumentInfo]:
        pass

    @abstractmethod
    def count(self, session_id: str) -> int:
        pass

    @abstractmethod
    def get_document_chunks(self, session_id: str, document_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_total_upload_bytes(self, session_id: str) -> int:
        pass

    @abstractmethod
    def touch_session(self, session_id: str) -> None:
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def cleanup_idle_sessions(self, ttl_minutes: int) -> int:
        pass

    @abstractmethod
    def cleanup_leftover_collections(self) -> int:
        pass

    @abstractmethod
    def get_active_session_count(self) -> int:
        pass


class ChromaVectorStore(BaseVectorStore):
    """Session-scoped vector store backed by ChromaDB and in-memory session metadata."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
    ):
        self.persist_dir = persist_dir
        self._sessions: Dict[str, SessionState] = {}

        os.makedirs(persist_dir, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB persistent client initialized at '{persist_dir}'")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreNotReadyError(f"Could not initialize ChromaDB: {str(e)}")

    def _collection_name(self, session_id: str) -> str:
        # Chroma collection names: letters, numbers, underscores, hyphens, 3-63 chars
        clean_id = session_id.replace("-", "_")
        return f"session_{clean_id}"

    def _get_collection(self, session_id: str):
        name = self._collection_name(session_id)
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def touch_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc)
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                created_at=now,
                last_active=now,
                documents={},
            )
        else:
            self._sessions[session_id].last_active = now

    def add_chunks(
        self,
        session_id: str,
        doc_info: DocumentInfo,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        if not chunks:
            return

        self.touch_session(session_id)

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.document_id,
                "doc_name": chunk.document_name,
                "page": chunk.page if chunk.page is not None else -1,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in chunks
        ]

        col = self._get_collection(session_id)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Update in-memory session registry
        self._sessions[session_id].documents[doc_info.id] = doc_info.model_dump()

    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[SearchResult]:
        self.touch_session(session_id)
        col_name = self._collection_name(session_id)

        try:
            col = self.client.get_collection(name=col_name)
            total = col.count()
            if total == 0:
                return []
        except Exception:
            return []

        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Vector search failed for session {session_id}: {e}")
            raise VectorStoreNotReadyError(f"Vector search failed: {str(e)}")

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, chunk_id in enumerate(ids):
            text = docs[i]
            meta = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else 1.0
            score = round(max(0.0, 1.0 - dist), 4)

            page_val = meta.get("page", -1)
            page = page_val if page_val != -1 else None

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=meta.get("doc_id", ""),
                    document_name=meta.get("doc_name", ""),
                    text=text,
                    score=score,
                    page=page,
                    chunk_index=meta.get("chunk_index", 0),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                )
            )

        return search_results

    def delete_document(self, session_id: str, document_id: str) -> bool:
        self.touch_session(session_id)
        s_state = self._sessions.get(session_id)
        if not s_state or document_id not in s_state.documents:
            return False

        col_name = self._collection_name(session_id)
        try:
            col = self.client.get_collection(name=col_name)
            col.delete(where={"doc_id": document_id})
        except Exception as e:
            logger.warning(f"Chroma delete by where clause encountered: {e}")

        del s_state.documents[document_id]
        logger.info(f"Deleted document {document_id} from session {session_id}")
        return True

    def get_document(self, session_id: str, document_id: str) -> Optional[DocumentInfo]:
        self.touch_session(session_id)
        s_state = self._sessions.get(session_id)
        if not s_state:
            return None
        doc_dict = s_state.documents.get(document_id)
        if doc_dict:
            return DocumentInfo(**doc_dict)
        return None

    def has_document_by_hash(self, session_id: str, content_hash: str) -> bool:
        self.touch_session(session_id)
        s_state = self._sessions.get(session_id)
        if not s_state:
            return False
        return any(doc.get("content_hash") == content_hash for doc in s_state.documents.values())

    def list_documents(self, session_id: str) -> List[DocumentInfo]:
        self.touch_session(session_id)
        s_state = self._sessions.get(session_id)
        if not s_state:
            return []
        return [DocumentInfo(**doc) for doc in s_state.documents.values()]

    def count(self, session_id: str) -> int:
        col_name = self._collection_name(session_id)
        try:
            col = self.client.get_collection(name=col_name)
            return col.count()
        except Exception:
            return 0

    def get_document_chunks(self, session_id: str, document_id: str) -> List[Dict[str, Any]]:
        self.touch_session(session_id)
        col_name = self._collection_name(session_id)
        try:
            col = self.client.get_collection(name=col_name)
            results = col.get(
                where={"doc_id": document_id},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error(f"Failed to fetch chunks for doc {document_id} in session {session_id}: {e}")
            return []

        chunks: List[Dict[str, Any]] = []
        if not results or not results.get("ids"):
            return chunks

        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        for i, chunk_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            page_val = meta.get("page", -1)
            page = page_val if page_val is not None and page_val != -1 else None
            chunk_index = meta.get("chunk_index", 0)
            chunks.append({
                "chunk_id": chunk_id,
                "index": chunk_index,
                "page": page,
                "text": text,
                "char_count": len(text),
            })

        chunks.sort(key=lambda x: x["index"])
        return chunks

    def get_total_upload_bytes(self, session_id: str) -> int:
        s_state = self._sessions.get(session_id)
        if not s_state:
            return 0
        return sum(d.get("file_size_bytes", 0) for d in s_state.documents.values())

    def delete_session(self, session_id: str) -> bool:
        col_name = self._collection_name(session_id)
        deleted = False

        try:
            self.client.delete_collection(name=col_name)
            deleted = True
        except Exception:
            pass

        if session_id in self._sessions:
            del self._sessions[session_id]
            deleted = True

        return deleted

    def cleanup_idle_sessions(self, ttl_minutes: int) -> int:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=ttl_minutes)
        idle_sessions = [
            sid for sid, s in self._sessions.items()
            if s.last_active < cutoff
        ]
        deleted_count = 0
        for sid in idle_sessions:
            logger.info(f"Session {sid} exceeded idle TTL ({ttl_minutes} min), cleaning up")
            if self.delete_session(sid):
                deleted_count += 1
        return deleted_count

    def cleanup_leftover_collections(self) -> int:
        deleted = 0
        try:
            collections = self.client.list_collections()
            for col in collections:
                cname = col.name if hasattr(col, "name") else str(col)
                if cname.startswith("session_"):
                    try:
                        self.client.delete_collection(name=cname)
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Could not delete leftover collection {cname}: {e}")
            self._sessions.clear()
            logger.info(f"Startup cleanup removed {deleted} leftover session collections")
        except Exception as e:
            logger.error(f"Error during startup collection cleanup: {e}")
        return deleted

    def get_active_session_count(self) -> int:
        return len(self._sessions)
