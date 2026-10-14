import io
import time
import uuid
import pytest
from fastapi.testclient import TestClient


def test_missing_or_invalid_session_id_400(test_app_and_state):
    client, _ = test_app_and_state
    # Raw client without default X-Session-ID header
    raw_client = TestClient(client.app)

    # 1. Missing header on GET /documents
    resp_missing = raw_client.get("/documents")
    assert resp_missing.status_code == 400
    assert resp_missing.json()["error"]["code"] == "INVALID_SESSION_ID"

    # 2. Invalid UUID format on GET /documents
    resp_invalid = raw_client.get("/documents", headers={"X-Session-ID": "invalid-uuid-123"})
    assert resp_invalid.status_code == 400
    assert resp_invalid.json()["error"]["code"] == "INVALID_SESSION_ID"

    # 3. Missing header on POST /ask
    resp_ask = raw_client.post("/ask", json={"question": "Test?"})
    assert resp_ask.status_code == 400
    assert resp_ask.json()["error"]["code"] == "INVALID_SESSION_ID"

    # 4. Missing header on POST /documents
    resp_upload = raw_client.post(
        "/documents",
        files=[("files", ("test.txt", io.BytesIO(b"sample text"), "text/plain"))],
    )
    assert resp_upload.status_code == 400
    assert resp_upload.json()["error"]["code"] == "INVALID_SESSION_ID"


def test_session_isolation(test_app_and_state, sample_markdown):
    client, _ = test_app_and_state
    raw_client = TestClient(client.app)

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    headers_a = {"X-Session-ID": session_a}
    headers_b = {"X-Session-ID": session_b}

    # Upload document to Session A
    files_a = [
        ("files", ("policy_a.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    resp_upload = raw_client.post("/documents", headers=headers_a, files=files_a)
    assert resp_upload.status_code == 201
    doc_id_a = resp_upload.json()["files"][0]["document_id"]
    assert doc_id_a is not None

    # Verify Session A can see document
    docs_a = raw_client.get("/documents", headers=headers_a).json()["documents"]
    assert len(docs_a) == 1
    assert docs_a[0]["filename"] == "policy_a.md"

    # Verify Session B cannot see Session A's document
    docs_b = raw_client.get("/documents", headers=headers_b).json()["documents"]
    assert len(docs_b) == 0

    # Verify Session B cannot fetch chunks of Session A's document
    chunks_b = raw_client.get(f"/documents/{doc_id_a}/chunks", headers=headers_b)
    assert chunks_b.status_code == 404
    assert chunks_b.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # Verify Session B cannot delete Session A's document
    del_b = raw_client.delete(f"/documents/{doc_id_a}", headers=headers_b)
    assert del_b.status_code == 404
    assert del_b.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # Query in Session A returns grounded answer
    ask_a = raw_client.post(
        "/ask",
        headers=headers_a,
        json={"question": "What are the storage tiers in Acme Cloud Storage?"},
    )
    assert ask_a.status_code == 200
    assert ask_a.json()["answerable"] is True

    # Query in Session B is refused (0 documents in Session B)
    ask_b = raw_client.post(
        "/ask",
        headers=headers_b,
        json={"question": "What are the storage tiers in Acme Cloud Storage?"},
    )
    assert ask_b.status_code == 200
    assert ask_b.json()["answerable"] is False
    assert len(ask_b.json()["sources"]) == 0


def test_session_delete_endpoint_idempotent(test_app_and_state, sample_markdown):
    client, _ = test_app_and_state
    raw_client = TestClient(client.app)
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}

    # Upload document to session
    files = [
        ("files", ("test.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    raw_client.post("/documents", headers=headers, files=files)

    # First delete -> returns 200 deleted=True
    del_1 = raw_client.delete("/session", headers=headers)
    assert del_1.status_code == 200
    assert del_1.json()["deleted"] is True
    assert del_1.json()["session_id"] == session_id

    # Second delete (now empty/deleted) -> returns 200 deleted=False
    del_2 = raw_client.delete("/session", headers=headers)
    assert del_2.status_code == 200
    assert del_2.json()["deleted"] is False

    # Documents list is empty
    docs = raw_client.get("/documents", headers=headers).json()["documents"]
    assert len(docs) == 0


def test_session_end_post_beacon(test_app_and_state, sample_markdown):
    client, _ = test_app_and_state
    raw_client = TestClient(client.app)
    session_id = str(uuid.uuid4())

    # Upload document
    files = [
        ("files", ("test.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    raw_client.post("/documents", headers={"X-Session-ID": session_id}, files=files)

    # Call POST /session/end?session_id=<uuid> simulating navigator.sendBeacon
    resp_beacon = raw_client.post(f"/session/end?session_id={session_id}")
    assert resp_beacon.status_code == 200
    assert resp_beacon.json()["deleted"] is True
    assert resp_beacon.json()["session_id"] == session_id

    # Second beacon call is safe and idempotent
    resp_beacon_again = raw_client.post(f"/session/end?session_id={session_id}")
    assert resp_beacon_again.status_code == 200
    assert resp_beacon_again.json()["deleted"] is False


def test_session_ttl_cleanup(test_app_and_state, sample_markdown):
    client, state = test_app_and_state
    raw_client = TestClient(client.app)
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}

    # Upload document
    files = [
        ("files", ("expire.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    raw_client.post("/documents", headers=headers, files=files)

    # Simulate session aging beyond TTL
    from datetime import datetime, timezone, timedelta
    store = state.vector_store
    assert session_id in store._sessions
    store._sessions[session_id].last_active = datetime.now(timezone.utc) - timedelta(minutes=60)

    # Run cleanup with 30 min TTL
    cleaned = store.cleanup_idle_sessions(ttl_minutes=30)
    assert cleaned == 1
    assert session_id not in store._sessions


def test_session_file_count_limit_413(test_app_and_state):
    client, state = test_app_and_state
    raw_client = TestClient(client.app)
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}

    # Set temporary low limit for testing
    state.settings.max_files_per_session = 2

    files_1 = [
        ("files", ("f1.txt", io.BytesIO(b"First file content"), "text/plain")),
        ("files", ("f2.txt", io.BytesIO(b"Second file content"), "text/plain")),
    ]
    resp_1 = raw_client.post("/documents", headers=headers, files=files_1)
    assert resp_1.status_code == 201

    # Attempt to upload 3rd file in same session
    files_2 = [
        ("files", ("f3.txt", io.BytesIO(b"Third file exceeds limit"), "text/plain")),
    ]
    resp_2 = raw_client.post("/documents", headers=headers, files=files_2)
    assert resp_2.status_code == 413
    assert resp_2.json()["error"]["code"] == "SESSION_LIMIT_EXCEEDED"


def test_session_total_size_limit_413(test_app_and_state):
    client, state = test_app_and_state
    raw_client = TestClient(client.app)
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}

    # Set temporary low limit of 0.001 MB (~1KB)
    state.settings.max_total_upload_mb = 0.001

    # 2KB file exceeds 1KB session limit
    large_content = b"A" * 2048
    files = [
        ("files", ("large.txt", io.BytesIO(large_content), "text/plain")),
    ]
    resp = raw_client.post("/documents", headers=headers, files=files)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "SESSION_LIMIT_EXCEEDED"
