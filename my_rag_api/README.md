# FastAPI RAG API

This project implements the four required endpoints in `my_rag_api.py`. It includes six example documents, persistent ChromaDB storage, Ollama embeddings and generation, Pydantic schemas, CORS, distance filtering, confidence labels, and dependency error handling.

## 1. Set up Python in PowerShell

Extract `rag-api.zip`. Open PowerShell in the extracted `rag-api` folder, where `my_rag_api.py` and `requirements.txt` are located. You can also open that folder in VS Code and use its terminal.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the Python you already have installed. They do not require the `py -3.12` command. Calling the environment's Python directly also avoids PowerShell activation-policy problems.

## 2. Start Ollama and install the models

Open Ollama from the Windows Start menu and leave it running. In PowerShell:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
ollama list
```

If PowerShell says `ollama` is not recognized, close and reopen PowerShell. If it still cannot find the command, use the executable from the default Windows installation folder:

```powershell
$ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
& $ollamaExe pull llama3.2
& $ollamaExe pull nomic-embed-text
& $ollamaExe list
```

If that executable is not present, locate your actual Ollama installation and use its path. You can verify that the server is responding with:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags"
```

If it refuses the connection, run `ollama serve` in a separate PowerShell window and leave that window open. With the default executable path, the equivalent command is:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

The API uses `nomic-embed-text` to embed both documents and questions, and `llama3.2` to generate answers. No OpenAI API key is needed.

## 3. Start the FastAPI server

From the project folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn my_rag_api:app --reload
```

Leave that terminal running. Open **http://127.0.0.1:8000/docs** in your browser. Each endpoint has a **Try it out** button. Click it, enter the request body if needed, then click **Execute**.

## 4. Test all endpoints in Swagger UI

| Test | Endpoint and body | Expected result |
| --- | --- | --- |
| Dependency health | `GET /health` | HTTP 200, `status: "healthy"`, both dependencies `ok: true`, and no missing models. |
| Empty database | Before ingestion: `POST /ask` with `{"question": "What is RAG?"}` | HTTP 200, a message to ingest documents first, empty sources, and low confidence. |
| Document ingestion | `POST /ingest` with `{}` | HTTP 200, `documents_loaded: 6` and `chunks_ingested: 12` for the included unchanged documents. |
| Document statistics | `GET /stats` | HTTP 200, `document_count: 6`, `chunk_count: 12`, and the two model names. |
| Grounded answer | `POST /ask` with `{"question": "What is RAG?"}` | HTTP 200, an answer based on context, sources, confidence, and retrieved chunks. The prompt requests numbered citations such as `[1]`. |
| Another answer | `POST /ask` with `{"question": "What does a Python virtual environment do?"}` | An answer grounded in `python.txt`; other relevant sources may also be retrieved. |
| Empty question | `POST /ask` with `{"question": ""}` | HTTP 422 with a Pydantic validation error. |
| Whitespace question | `POST /ask` with `{"question": "   "}` | HTTP 422. |
| Repeated ingestion | `POST /ingest` again with `{}`, then `GET /stats` | Counts stay the same; chunks are updated without duplication. |
| Ollama unavailable | After successful ingestion, quit Ollama/stop `ollama serve`, then call `POST /ask` | HTTP 503 with a clear Ollama message. `GET /health` also returns HTTP 503. |

For a question outside the documents, try `{"question": "What will the weather be next month?"}`. Depending on the retrieved distances, the API will either skip generation with a no-relevant-information message or ask the model to answer only from context and acknowledge that the answer is unavailable. A distance threshold is a relevance heuristic; it does not prove that a question is answerable.

An optional request can override the defaults:

```json
{
  "question": "What is RAG?",
  "top_k": 3,
  "distance_threshold": 0.75,
  "temperature": 0
}
```

With an empty collection, `/ask` returns the ingestion message without contacting Ollama. `/stats` also works independently of Ollama. `/health` checks both dependencies and reports missing models; a running server with a missing required model is considered unhealthy.

## 5. Configuration

Set environment variables in the same PowerShell window before starting Uvicorn. Restart the API after changing them.

```powershell
$env:RAG_TOP_K = "3"
$env:RAG_DISTANCE_THRESHOLD = "1.0"
$env:RAG_TEMPERATURE = "0.1"
$env:RAG_OLLAMA_MODEL = "llama3.2"
$env:RAG_EMBEDDING_MODEL = "nomic-embed-text"
$env:RAG_OLLAMA_URL = "http://127.0.0.1:11434"
$env:RAG_CORS_ORIGINS = '["http://localhost:3000","http://localhost:5173","http://localhost:8501"]'
.\.venv\Scripts\python.exe -m uvicorn my_rag_api:app --reload
```

Other settings include `RAG_DOCS_DIR`, `RAG_CHROMA_DIR`, `RAG_COLLECTION_NAME`, `RAG_MAX_CHUNK_CHARS`, `RAG_BATCH_SIZE`, `RAG_OLLAMA_TIMEOUT`, and `RAG_HEALTH_TIMEOUT`. Paths default to folders beside the script. Add your own UTF-8 `.txt` or `.md` files under `docs/`, then call `/ingest` again.

The default collection is `my_rag_api`, separate from earlier exercises. A successful ingestion synchronizes this dedicated collection with the current nonempty document folder, including removing stale chunks. If you change the embedding model, choose a new collection name and ingest again; incompatible collections return HTTP 409. Ollama embedding failures are detected before database writes begin. ChromaDB does not provide an atomic transaction across this script's multiple write batches, so a database failure during writing may leave a partially updated collection; rerun ingestion after fixing the failure.

Only chunks with cosine distance strictly below the threshold are supplied to the model. Confidence is `high` below 0.5, `medium` from 0.5 to below 1.0, and `low` at 1.0 or above. Empty or fully filtered results return low confidence. Confidence measures retrieval similarity, not the factual correctness of the generated answer. The `sources` list identifies the documents supplied as context; the prompt instructs the model to cite them, but citations are not independently verified.

## 6. Automated tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use real persistent ChromaDB in temporary folders and simulated Ollama HTTP responses. They cover all four routes, OpenAPI schemas, blank questions, ingestion updates, source filtering, confidence boundaries, configuration, CORS, timeouts, unavailable dependencies, and malformed model responses. They do not require a downloaded model or prove a live model's answer quality.

Verification: 44 automated tests passed on Linux with Python 3.12. The included document collection was also checked: all four routes returned HTTP 200, ingestion produced six documents and twelve chunks, and a blank question returned HTTP 422. Windows/Python 3.14 wheel availability was checked for the main dependency graph, but the API was not run on Windows. This environment's browser blocked localhost pages, so the Swagger UI click-through tests above must be completed locally with your running Ollama models.

## Official documentation

- [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/)
- [FastAPI response models](https://fastapi.tiangolo.com/tutorial/response-model/)
- [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/)
- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [Ollama embeddings](https://docs.ollama.com/api/embed)
- [Ollama chat](https://docs.ollama.com/api/chat)
- [Ollama model listing](https://docs.ollama.com/api/tags)
- [Ollama on Windows](https://docs.ollama.com/windows)
