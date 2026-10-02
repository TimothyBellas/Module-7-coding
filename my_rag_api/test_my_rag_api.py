"""Route tests with real persistent ChromaDB and a simulated Ollama HTTP API.

These check API behavior, not a real model's answer quality.
Run: python -m pytest -q
"""

import json
import math
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import my_rag_api
from my_rag_api import Settings, confidence_for, create_app


class FakeOllama:
    def __init__(self):
        self.offline = False
        self.timeout = False
        self.missing_chat_model = False
        self.empty_answer = False
        self.bad_embeddings = None
        self.invalid_json = False
        self.fail_embed_call = None
        self.embed_calls = 0
        self.models = ["llama3.2:latest", "nomic-embed-text:latest"]
        self.requests = []

    @staticmethod
    def vector(text):
        text = text.lower()
        if "medium query" in text:
            return [0.1, 0.3, 0.0, math.sqrt(0.9)]
        if "rag" in text or "retrieval" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "python" in text or "virtual" in text:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def __call__(self, request):
        payload = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, payload))
        if self.offline:
            raise httpx.ConnectError("Connection refused", request=request)
        if self.timeout:
            raise httpx.ReadTimeout("Timed out", request=request)
        if self.invalid_json:
            return httpx.Response(200, text="not JSON")
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/embed":
            self.embed_calls += 1
            if self.embed_calls == self.fail_embed_call:
                return httpx.Response(503, json={"error": "server unavailable"})
            if self.bad_embeddings is not None:
                return httpx.Response(200, json={"embeddings": self.bad_embeddings})
            return httpx.Response(200, json={"embeddings": [self.vector(t) for t in payload["input"]]})
        if request.url.path == "/api/chat":
            if self.missing_chat_model:
                return httpx.Response(404, json={"error": "model not found"})
            context = json.loads(payload["messages"][1]["content"])["context"]
            answer = "" if self.empty_answer else f"{context[0]['text']} {context[0]['label']}"
            return httpx.Response(200, json={"message": {"role": "assistant", "content": answer}})
        return httpx.Response(404)


@pytest.fixture
def api(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "python.txt").write_text(
        "Python is a programming language.\n\nA Python virtual environment isolates dependencies.", encoding="utf-8"
    )
    (docs / "rag.txt").write_text(
        "RAG means Retrieval-Augmented Generation and uses retrieved documents as context.", encoding="utf-8"
    )
    settings = Settings(docs_dir=docs, chroma_dir=tmp_path / "chroma", collection_name="test_rag_api")
    ollama = FakeOllama()
    app = create_app(settings, http_transport=httpx.MockTransport(ollama))
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, ollama=ollama, docs=docs, settings=settings)


def ingest(api, **body):
    response = api.client.post("/ingest", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_swagger_and_openapi_expose_all_four_routes_and_models(api):
    assert "swagger-ui" in api.client.get("/docs").text
    spec = api.client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/ask", "/ingest", "/stats", "/health"}
    for path, method, model in [
        ("/ask", "post", "AskResponse"), ("/ingest", "post", "IngestResponse"),
        ("/stats", "get", "StatsResponse"), ("/health", "get", "HealthResponse"),
    ]:
        schema = spec["paths"][path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/" + model)
    assert "422" in spec["paths"]["/ask"]["post"]["responses"]


def test_health_checks_chroma_ollama_and_models(api):
    response = api.client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["chromadb"]["ok"] and data["ollama"]["ok"]
    assert data["missing_models"] == []


def test_health_returns_503_when_ollama_is_offline(api):
    api.ollama.offline = True
    response = api.client.get("/health")
    assert response.status_code == 503
    assert response.json()["chromadb"]["ok"] is True
    assert response.json()["ollama"]["ok"] is False


def test_health_reports_missing_embedding_model(api):
    api.ollama.models = ["llama3.2:latest"]
    response = api.client.get("/health")
    assert response.status_code == 503
    assert response.json()["ollama"]["ok"] is True
    assert response.json()["missing_models"] == ["nomic-embed-text"]


def test_ask_before_ingest_returns_clear_response_without_using_ollama(api):
    api.ollama.offline = True
    response = api.client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert "No documents have been ingested" in data["answer"]
    assert data["sources"] == [] and data["confidence"] == "low"
    assert data["chunks_retrieved"] == 0
    assert api.ollama.requests == []


@pytest.mark.parametrize("body", [
    {"question": ""}, {"question": "   "}, {"question": "\n\t"},
    {"question": None}, {}, {"question": 123}, {"question": "x" * 2001},
    {"question": "test", "top_k": 0}, {"question": "test", "top_k": 21},
    {"question": "test", "distance_threshold": 0},
    {"question": "test", "temperature": 2},
    {"question": "test", "unknown_option": True},
])
def test_invalid_requests_return_pydantic_422_without_ollama_calls(api, body):
    assert api.client.post("/ask", json=body).status_code == 422
    assert api.ollama.requests == []


def test_ingest_and_stats_distinguish_files_from_chunks_and_are_repeatable(api):
    first = ingest(api)
    assert first["documents_loaded"] == 2 and first["chunks_ingested"] == 3
    assert ingest(api)["total_chunks"] == 3
    response = api.client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 2 and data["chunk_count"] == 3
    assert data["ollama_model"] == "llama3.2"
    assert data["embedding_model"] == "nomic-embed-text"
    assert data["distance_metric"] == "cosine"


def test_ingest_updates_edited_files_and_removes_stale_chunks(api):
    ingest(api)
    (api.docs / "python.txt").write_text("Python documentation was updated.", encoding="utf-8")
    assert ingest(api)["total_chunks"] == 2
    (api.docs / "python.txt").unlink()
    assert ingest(api)["total_chunks"] == 1
    assert api.client.get("/stats").json()["document_count"] == 1


def test_ask_returns_filtered_context_sources_and_confidence(api):
    ingest(api)
    response = api.client.post("/ask", json={"question": "  What is RAG?  "})
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == ["rag.txt"] and data["confidence"] == "high"
    assert data["chunks_retrieved"] == 1 and "[1]" in data["answer"]
    assert data["best_distance"] == pytest.approx(0.0)
    payload = [p for path, p in api.ollama.requests if path == "/api/chat"][-1]
    assert payload["stream"] is False
    assert "Never invent facts" in payload["messages"][0]["content"]
    prompt = json.loads(payload["messages"][1]["content"])
    assert prompt["question"] == "What is RAG?"
    assert len(prompt["context"]) == 1
    assert prompt["context"][0]["text"] == data["retrieved_chunks"][0]["text"]


def test_per_request_parameters_and_zero_temperature_work(api):
    ingest(api)
    response = api.client.post("/ask", json={
        "question": "What is Python?", "top_k": 1, "distance_threshold": 0.05, "temperature": 0,
    })
    assert response.status_code == 200
    assert response.json()["chunks_retrieved"] == 1
    payload = [p for path, p in api.ollama.requests if path == "/api/chat"][-1]
    assert payload["options"]["temperature"] == 0


def test_no_qualifying_chunks_skips_generation_and_returns_low_confidence(api):
    ingest(api)
    response = api.client.post("/ask", json={"question": "Who will win an unrelated sports game?"})
    assert response.status_code == 200
    assert "No relevant information" in response.json()["answer"]
    assert response.json()["sources"] == []
    assert response.json()["confidence"] == "low"
    assert response.json()["best_distance"] == pytest.approx(1.0)
    assert not any(path == "/api/chat" for path, _ in api.ollama.requests)


def test_medium_confidence_and_low_confidence_with_relaxed_threshold(api):
    ingest(api)
    medium = api.client.post("/ask", json={"question": "medium query"})
    assert medium.json()["confidence"] == "medium"
    low = api.client.post("/ask", json={"question": "unrelated", "distance_threshold": 1.5})
    assert low.status_code == 200 and low.json()["confidence"] == "low"


@pytest.mark.parametrize("distance, expected", [
    (0.0, "high"), (0.4999, "high"), (0.5, "medium"),
    (0.9999, "medium"), (1.0, "low"), (1.5, "low"),
])
def test_required_confidence_boundaries(distance, expected):
    assert confidence_for(distance) == expected


def test_ask_and_ingest_return_503_when_ollama_stops(api):
    ingest(api)
    api.ollama.offline = True
    assert api.client.post("/ask", json={"question": "What is RAG?"}).status_code == 503
    assert api.client.post("/ingest", json={}).status_code == 503
    assert api.client.get("/stats").json()["chunk_count"] == 3


def test_missing_generation_model_returns_503_and_clear_message(api):
    ingest(api)
    api.ollama.missing_chat_model = True
    response = api.client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 503
    assert "ollama pull llama3.2" in response.json()["detail"]


def test_timeout_returns_503(api):
    api.ollama.timeout = True
    response = api.client.post("/ingest", json={})
    assert response.status_code == 503 and "timed out" in response.json()["detail"]


@pytest.mark.parametrize("bad_vectors", [[], [[]], [[0, 0, 0, 0]], [[1, 0, 0, 0]]])
def test_bad_upstream_embeddings_return_502(api, bad_vectors):
    api.ollama.bad_embeddings = bad_vectors
    assert api.client.post("/ingest", json={}).status_code == 502
    assert api.client.get("/stats").json()["chunk_count"] == 0


def test_invalid_upstream_json_returns_502(api):
    api.ollama.invalid_json = True
    assert api.client.post("/ingest", json={}).status_code == 502


def test_empty_generation_answer_returns_502(api):
    ingest(api)
    api.ollama.empty_answer = True
    assert api.client.post("/ask", json={"question": "What is RAG?"}).status_code == 502


def test_embedding_failure_preserves_existing_collection(api):
    ingest(api)
    (api.docs / "extra.txt").write_text("Extra Python documentation.", encoding="utf-8")
    api.ollama.fail_embed_call = api.ollama.embed_calls + 2
    assert api.client.post("/ingest", json={"batch_size": 1}).status_code == 503
    assert api.client.get("/stats").json()["chunk_count"] == 3


def test_chroma_initialization_failure_keeps_health_endpoint_available(api, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("private database error")
    monkeypatch.setattr(my_rag_api.chromadb, "PersistentClient", fail)
    response = api.client.get("/health")
    assert response.status_code == 503
    assert response.json()["chromadb"]["ok"] is False
    assert response.json()["ollama"]["ok"] is True
    stats = api.client.get("/stats")
    assert stats.status_code == 503 and "private database error" not in stats.text


def test_collection_embedding_model_conflict_is_reported(api):
    ingest(api)
    changed = api.settings.model_copy(update={"embedding_model": "other-model"})
    app = create_app(changed, http_transport=httpx.MockTransport(api.ollama))
    with TestClient(app) as client:
        assert client.get("/stats").status_code == 409


def test_missing_or_empty_docs_returns_clear_400(api):
    for file in api.docs.iterdir():
        file.unlink()
    assert api.client.post("/ingest", json={}).status_code == 400
    api.docs.rmdir()
    response = api.client.post("/ingest", json={})
    assert response.status_code == 400 and "missing" in response.json()["detail"]


def test_cors_preflight_and_validation_responses(api):
    origin = "http://localhost:5173"
    response = api.client.options("/ask", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    invalid = api.client.post("/ask", json={"question": ""}, headers={"Origin": origin})
    assert invalid.status_code == 422
    assert invalid.headers["access-control-allow-origin"] == origin


def test_settings_accept_environment_overrides(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_DISTANCE_THRESHOLD", "0.75")
    monkeypatch.setenv("RAG_CORS_ORIGINS", '["http://localhost:9000"]')
    settings = Settings()
    assert settings.top_k == 5 and settings.distance_threshold == 0.75
    assert settings.cors_origins == ["http://localhost:9000"]
