# Validation performed

Checked on October 7, 2026, using CPython **3.11.16** on Linux. All six direct dependencies in `requirements.txt` installed successfully, and their versions matched the file. Dependency resolution was also checked for Python 3.11 on Linux x86-64.

**12 checks passed.** ChromaDB used real persistent storage. Ollama embedding, model-list, and chat responses were mocked for the successful ingestion and question-answering checks. A separate live Uvicorn process was tested through actual HTTP requests.

| Check | Observed result |
| --- | --- |
| Uploaded API and syntax | Packaged `my_rag_api.py` matches the upload byte for byte; syntax is valid |
| Dockerfile and `.dockerignore` | Required base, system dependency instruction, dependency-first copies, pip flag, code/docs copies, port, Uvicorn command, and exclusions are present |
| Python dependencies | All six pinned direct dependencies installed and matched their listed versions |
| Swagger and OpenAPI | `/docs` and `/openapi.json` returned HTTP 200; all four requested endpoints are documented |
| Empty database and validation | `/ask` explained that no documents were ingested; empty and whitespace-only questions returned HTTP 422 |
| Dependency health | Real ChromaDB initialized; `/health` returned HTTP 200 when mocked Ollama listed both models |
| Ingestion | `/ingest` returned HTTP 200 and stored **6 documents / 18 chunks** in real ChromaDB using mocked embeddings |
| Question answering | `/ask` returned HTTP 200 with the retrieved `rag.txt` source, confidence, retrieved chunks, and a citation in the mocked answer |
| Statistics | `/stats` reported 6 documents, 18 chunks, and the expected chat and embedding model names |
| Re-ingestion | Running ingestion again kept the stored chunk count at 18 |
| Persistence and offline Ollama | A new app opened the saved collection; unavailable mocked Ollama caused HTTP 503 for `/health`, `/ask`, and `/ingest`; stored chunks remained intact |
| Live Uvicorn startup | The configured command bound to `0.0.0.0:8000`; actual HTTP requests to `/docs`, `/openapi.json`, and `/stats` returned 200. With host Ollama unreachable in this environment, `/health` returned 503 and reported accessible ChromaDB |

## Still requires your Docker installation

Docker was not available in this environment. These assignment steps remain **unverified**:

1. Building the actual `my-rag-api` image.
2. Starting it with `docker run -p 8000:8000 my-rag-api` and testing container-to-host Ollama connectivity.
3. Using Swagger UI in your browser to perform `/ingest` and `/ask` with real Ollama models and assess the generated answer.
4. Changing a Python comment, rebuilding, and observing **CACHED** for the pip installation layer.

The full PowerShell instructions are in `README.md`. Successful mock responses are not evidence of real model behavior, and static Dockerfile checks are not evidence of an actual image build or cache hit.
