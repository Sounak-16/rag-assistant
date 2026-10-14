import io


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True
    assert "document_count" in data
    assert "chunk_count" in data


def test_upload_success_and_deduplication(test_client, sample_markdown):
    files = [
        ("files", ("handbook.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    response = test_client.post("/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert len(data["files"]) == 1
    assert data["files"][0]["filename"] == "handbook.md"
    assert data["files"][0]["status"] == "indexed"
    assert data["files"][0]["total_chunks"] > 0
    assert data["files"][0]["preview"] is not None
    assert data["files"][0]["preview"] == sample_markdown[:300]
    assert len(data["uploaded"]) == 1
    assert len(data["skipped"]) == 0

    # Re-upload the exact same file (should be marked duplicate)
    files_again = [
        ("files", ("handbook.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    resp_again = test_client.post("/documents", files=files_again)
    assert resp_again.status_code == 201
    data_again = resp_again.json()
    assert len(data_again["files"]) == 1
    assert data_again["files"][0]["status"] == "duplicate"
    assert len(data_again["uploaded"]) == 0
    assert "handbook.md" in data_again["skipped"]


def test_upload_unsupported_media_type(test_client):
    files = [
        ("files", ("spreadsheet.xlsx", io.BytesIO(b"fake excel content"), "application/vnd.ms-excel")),
    ]
    response = test_client.post("/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert len(data["files"]) == 1
    assert data["files"][0]["status"] == "failed"
    assert "unsupported" in data["files"][0]["error"].lower()


def test_upload_empty_file_rejected(test_client):
    files = [
        ("files", ("empty.txt", io.BytesIO(b""), "text/plain")),
    ]
    response = test_client.post("/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert len(data["files"]) == 1
    assert data["files"][0]["status"] == "failed"
    assert "empty" in data["files"][0]["error"].lower()


def test_upload_batch_partial_failure_resilience(test_client, sample_markdown):
    files = [
        ("files", ("valid.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
        ("files", ("corrupt.unknown", io.BytesIO(b"bad binary"), "application/octet-stream")),
    ]
    response = test_client.post("/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert len(data["files"]) == 2
    assert data["files"][0]["filename"] == "valid.md"
    assert data["files"][0]["status"] == "indexed"
    assert data["files"][1]["filename"] == "corrupt.unknown"
    assert data["files"][1]["status"] == "failed"
    assert data["files"][1]["error"] is not None
    assert len(data["uploaded"]) == 1


def test_get_document_chunks_success(test_client, sample_markdown):
    # Upload doc
    files = [
        ("files", ("doc_chunks.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    res_upload = test_client.post("/documents", files=files)
    assert res_upload.status_code == 201
    doc_id = res_upload.json()["files"][0]["document_id"]
    assert doc_id is not None

    # Retrieve chunks
    res_chunks = test_client.get(f"/documents/{doc_id}/chunks")
    assert res_chunks.status_code == 200
    chunk_data = res_chunks.json()
    assert chunk_data["document_id"] == doc_id
    assert chunk_data["filename"] == "doc_chunks.md"
    assert chunk_data["total_chunks"] == len(chunk_data["chunks"])
    assert chunk_data["chunk_size"] > 0
    assert chunk_data["chunk_overlap"] >= 0
    assert len(chunk_data["chunks"]) > 0

    # Verify chunks are strictly ordered by index and fields are populated
    for i, chunk in enumerate(chunk_data["chunks"]):
        assert chunk["index"] == i
        assert chunk["chunk_id"].startswith(f"{doc_id}_")
        assert len(chunk["text"]) > 0
        assert chunk["char_count"] == len(chunk["text"])


def test_get_document_chunks_not_found(test_client):
    response = test_client.get("/documents/nonexistent_doc_id/chunks")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_document_list_and_delete(test_client):
    # Upload doc
    files = [
        ("files", ("notes.txt", io.BytesIO(b"Important engineering meeting notes."), "text/plain")),
    ]
    res_upload = test_client.post("/documents", files=files)
    assert res_upload.status_code == 201
    doc_id = res_upload.json()["files"][0]["document_id"]

    # List documents and verify document_id is present
    res_list = test_client.get("/documents")
    assert res_list.status_code == 200
    docs = res_list.json()["documents"]
    matched = [d for d in docs if d["id"] == doc_id]
    assert len(matched) == 1
    assert matched[0]["document_id"] == doc_id

    # Delete document
    res_del = test_client.delete(f"/documents/{doc_id}")
    assert res_del.status_code == 200
    assert res_del.json()["document_id"] == doc_id

    # Delete again -> 404
    res_del_404 = test_client.delete(f"/documents/{doc_id}")
    assert res_del_404.status_code == 404
    assert res_del_404.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_ask_answerable(test_client, sample_markdown):
    # Ingest markdown containing Acme storage policy
    files = [
        ("files", ("acme_policy.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    test_client.post("/documents", files=files)

    # Query for storage tiers
    payload = {"question": "What are the storage tiers in Acme Cloud Storage?", "top_k": 3}
    response = test_client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["answerable"] is True
    assert len(data["sources"]) > 0
    # Verify sources structure and real excerpt
    first_source = data["sources"][0]
    assert first_source["document"] == "acme_policy.md"
    assert "chunk_id" in first_source
    assert len(first_source["excerpt"]) > 0
    assert first_source["score"] > 0


def test_ask_unanswerable(test_client, sample_markdown):
    # Ingest document
    files = [
        ("files", ("acme_policy.md", io.BytesIO(sample_markdown.encode("utf-8")), "text/markdown")),
    ]
    test_client.post("/documents", files=files)

    # Ask completely unrelated question not in documents
    payload = {"question": "What was the score of the 1998 World Cup final football match?"}
    response = test_client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["answerable"] is False
    assert "couldn't find this information" in data["answer"].lower()
    assert data["sources"] == []


def test_ask_invalid_payload(test_client):
    # Empty string should fail validation
    response = test_client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
