"""Configuration behavior checks; no Docker, Ollama, or model download needed."""

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
import requests

from config import BACKEND_DIR, Settings
import main


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    for name in list(os.environ):
        if name.upper() in Settings.model_fields:
            monkeypatch.delenv(name)


def test_defaults_work_without_a_dotenv_file():
    value = Settings(_env_file=None)
    assert value.OLLAMA_URL == "http://localhost:11434"
    assert value.MODEL_NAME == "llama3.2:1b"
    assert value.CHROMA_PATH == BACKEND_DIR / "chroma_data"
    assert value.MAX_RESULTS == 3
    assert value.CONFIDENCE_THRESHOLD == 1.0
    assert value.DEBUG is False


def test_dotenv_parses_types_and_process_environment_takes_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "OLLAMA_URL=http://ollama:11434/\nMODEL_NAME=llama3.2:3b\n"
        "MAX_RESULTS=7\nCONFIDENCE_THRESHOLD=0.8\nDEBUG=true\n"
        'CORS_ORIGINS=["http://localhost:8501"]\nBACKEND_PORT=8001\n',
        encoding="utf-8-sig",  # Accept a BOM from Windows editors.
    )
    value = Settings(_env_file=env)
    assert value.OLLAMA_URL == "http://ollama:11434"
    assert value.MODEL_NAME == "llama3.2:3b"
    assert value.MAX_RESULTS == 7
    assert value.CONFIDENCE_THRESHOLD == 0.8
    assert value.DEBUG is True
    assert value.CORS_ORIGINS == ["http://localhost:8501"]
    monkeypatch.setenv("MODEL_NAME", "environment-model")
    assert Settings(_env_file=env).MODEL_NAME == "environment-model"


@pytest.mark.parametrize("name,value", [
    ("MAX_RESULTS", "0"),
    ("MAX_RESULTS", "not-a-number"),
    ("CONFIDENCE_THRESHOLD", "0"),
    ("CONFIDENCE_THRESHOLD", "nan"),
    ("DEBUG", "maybe"),
    ("OLLAMA_URL", "not-a-url"),
    ("MODEL_NAME", "   "),
])
def test_bad_environment_values_fail_clearly(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_new_api_process_reads_edited_project_dotenv(tmp_path, monkeypatch):
    # Stage actual backend modules so both process launches find their own .env.
    backend = tmp_path / "backend"
    backend.mkdir()
    for name in ["config.py", "main.py"]:
        shutil.copyfile(BACKEND_DIR / name, backend / name)
    monkeypatch.setenv("PYTHONPATH", str(backend))
    code = (
        "import json; from fastapi.testclient import TestClient; "
        "from main import app; print(json.dumps({"
        "'model': TestClient(app).get('/').json()['model'], 'debug': app.debug}))"
    )
    env_path = tmp_path / ".env"
    observed = []
    for model, debug in [("llama3.2:1b", "false"), ("llama3.2:3b", "true")]:
        env_path.write_text(f"MODEL_NAME={model}\nDEBUG={debug}\n", encoding="utf-8")
        child = subprocess.run(
            [sys.executable, "-c", code], cwd=backend,
            capture_output=True, text=True, check=True, timeout=60,
        )
        observed.append(json.loads(child.stdout.strip().splitlines()[-1]))
    assert observed == [
        {"model": "llama3.2:1b", "debug": False},
        {"model": "llama3.2:3b", "debug": True},
    ]


def test_configured_path_is_used_by_chroma_and_stats(tmp_path, monkeypatch):
    configured = Settings(
        _env_file=None, CHROMA_PATH=tmp_path / "database", MODEL_NAME="custom-model",
        MAX_RESULTS=5, CONFIDENCE_THRESHOLD=0.8,
    )
    monkeypatch.setattr(main, "settings", configured)
    monkeypatch.setattr(main.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200))
    with TestClient(main.app) as client:
        assert client.get("/").json()["model"] == "custom-model"
        stats = client.get("/stats").json()
        assert stats["chroma_path"] == str(tmp_path / "database")
        assert stats["max_results"] == 5
        assert stats["confidence_threshold"] == 0.8
        assert (tmp_path / "database" / "chroma.sqlite3").is_file()
        assert client.get("/health").json()["ollama"] == "connected"
        assert client.post("/ask", json={"question": "What is RAG?"}).json()["chunks_retrieved"] == 0
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.get("/docs").status_code == 200


class SearchCollection:
    def __init__(self, distances):
        self.distances = distances
        self.requested_count = None

    def count(self):
        return len(self.distances)

    def query(self, *, n_results, **kwargs):
        self.requested_count = n_results
        selected = self.distances[:n_results]
        return {
            "documents": [[f"Context {i}" for i in range(len(selected))]],
            "metadatas": [[{"source": f"source{i}.txt", "chunk": i} for i in range(len(selected))]],
            "distances": [selected],
        }


def test_generation_uses_model_url_limit_threshold_and_timeouts_from_settings(monkeypatch):
    configured = Settings(
        _env_file=None, MODEL_NAME="changed-model", OLLAMA_URL="http://service:11434/",
        MAX_RESULTS=2, CONFIDENCE_THRESHOLD=0.6, TEMPERATURE=0.3,
        OLLAMA_CONNECT_TIMEOUT=4, OLLAMA_TIMEOUT=25,
    )
    monkeypatch.setattr(main, "settings", configured)
    collection = SearchCollection([0.4, 0.6, 0.1])
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(
            status_code=200, raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "Grounded answer [source0.txt]."}},
        )

    monkeypatch.setattr(main.requests, "post", post)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(collection=collection)))
    answer = main.ask(main.AskRequest(question="What is RAG?"), request)
    assert collection.requested_count == 2
    assert answer.chunks_retrieved == 1  # Distance equal to the threshold is excluded.
    assert answer.sources == ["source0.txt"]
    assert answer.confidence == "medium"  # 0.4 is between threshold/2 and threshold.
    assert captured["url"] == "http://service:11434/api/chat"
    assert captured["json"]["model"] == "changed-model"
    assert captured["json"]["options"]["temperature"] == 0.3
    assert captured["timeout"] == (4, 25)


def test_filtered_context_does_not_call_ollama(monkeypatch):
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, CONFIDENCE_THRESHOLD=0.3))
    collection = SearchCollection([0.3, 1.0])
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(collection=collection)))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Ollama must not be called when no context passes the threshold.")

    monkeypatch.setattr(main.requests, "post", unexpected_call)
    answer = main.ask(main.AskRequest(question="Unknown topic"), request)
    assert answer.confidence == "low"
    assert answer.chunks_retrieved == 0


def test_unavailable_ollama_still_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, CHROMA_PATH=tmp_path / "database"))

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("service down")

    monkeypatch.setattr(main.requests, "get", unavailable)
    with TestClient(main.app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["ollama"] == "unavailable"
