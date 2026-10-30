from fastapi.testclient import TestClient


def test_healthz_and_readyz(client: TestClient) -> None:
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "database": True,
        "redis": True,
        "llm_provider": "fake",
    }


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    echoed = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["x-request-id"] == "abc-123"
    generated = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert generated.headers["x-request-id"] != "bad id with spaces"
    assert len(generated.headers["x-request-id"]) == 32


def test_conversation_crud(client: TestClient) -> None:
    created = client.post("/conversations", json={"title": "First"})
    assert created.status_code == 201
    body = created.json()
    assert body["title"] == "First"
    assert body["message_count"] == 0

    untitled = client.post("/conversations")
    assert untitled.status_code == 201
    assert untitled.json()["title"] is None

    fetched = client.get(f"/conversations/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]

    listing = client.get("/conversations", params={"limit": 1})
    assert listing.status_code == 200
    assert listing.json()["total"] == 2
    assert len(listing.json()["items"]) == 1

    history = client.get(f"/conversations/{body['id']}/messages")
    assert history.status_code == 200
    assert history.json() == {"conversation_id": body["id"], "items": []}


def test_unknown_conversation_is_404(client: TestClient) -> None:
    assert client.get("/conversations/does-not-exist").status_code == 404
    assert client.get("/conversations/does-not-exist/messages").status_code == 404


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/conversations", json={"title": "x" * 201}).status_code == 422
    assert client.get("/conversations", params={"limit": 0}).status_code == 422
    assert client.post("/documents", json={"source": "a.md", "content": ""}).status_code == 422


def test_document_ingest_and_search(client: TestClient) -> None:
    ingested = client.post(
        "/documents",
        json={
            "source": "ops.md",
            "content": "# Operations\n## Locking\nEach conversation has a Redis lock so turns "
            "never interleave.\n## Storage\nPostgres keeps the full message history.",
        },
    )
    assert ingested.status_code == 201
    assert ingested.json()["chunks"] == 2

    results = client.get("/documents/search", params={"q": "redis lock turns", "k": 2})
    assert results.status_code == 200
    top = results.json()["hits"][0]
    assert top["source"] == "ops.md"
    assert top["heading"] == "Operations > Locking"
