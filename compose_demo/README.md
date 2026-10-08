# FastAPI + Ollama Docker Compose assignment

This project builds a FastAPI RAG backend and runs Ollama in a second container.
The backend reaches Ollama at `http://ollama:11434` on the Compose network.
Named volumes hold the ChromaDB database and Ollama models.

## Included files

| Path | Purpose |
| --- | --- |
| `docker-compose.yml` | Two services, ports, environment variables, startup checks, and named volumes |
| `.env` | Host ports and generation model |
| `backend/Dockerfile` | Python 3.11 image, dependency caching, application files, and Uvicorn command |
| `backend/main.py` | Root, health, ingestion, statistics, and RAG question endpoints |
| `backend/requirements.txt` | Compatible, pinned application dependencies |
| `backend/.dockerignore` | Excludes local environments and databases from the image |
| `backend/docs/sample.txt` | Four paragraphs for ingestion and the persistence test |
| `verify.ps1` | Windows verification with a saved transcript of actual results |

The backend uses Chroma's default `all-MiniLM-L6-v2` embedding model. Ollama uses
`llama3.2:1b` to generate answers. The first ingestion downloads Chroma's embedding
model and needs internet access. Ollama's generation model is downloaded separately.

## Simplest way to run and verify on Windows

1. Extract the ZIP and open the `compose-demo` folder in VS Code.
2. Open Docker Desktop and wait until its engine is running. Use Linux containers.
3. Open a PowerShell terminal in that folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
```

This command builds and starts the containers, checks `/` and `/health`, ingests
the sample document, records a nonzero count, runs `docker compose down`, restarts
the containers, and verifies that the same count remains **without re-ingesting**.
It leaves the containers running and saves the actual output in
`verification-results.txt`. Keep this output as assignment evidence.

For a complete RAG generation test and Ollama model persistence check, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1 -TestRag
```

The additional test pulls the configured model inside the Ollama container,
submits a question to `/ask`, and compares model digests before and after restart.
Model download and generation may take several minutes on a CPU.

Open [Swagger UI](http://localhost:8000/docs) to try the API yourself. If you change
`BACKEND_PORT` in `.env`, use that port in the browser URL.

## Manual PowerShell verification

Run these commands from the `compose-demo` folder. These examples use the default
ports from `.env`. Modern Docker Desktop uses `docker compose` with a space;
it performs the same assignment operations as the older `docker-compose` spelling.

Build and start:

```powershell
docker info
docker compose config --quiet
docker compose up --build -d --wait --wait-timeout 180
docker compose ps
Invoke-RestMethod http://localhost:8000/
Invoke-RestMethod http://localhost:8000/health
```

Expected root response:

```json
{
  "message": "RAG API running in Docker",
  "model": "llama3.2:1b"
}
```

Expected health response on a fresh database:

```json
{
  "status": "healthy",
  "chromadb": "accessible",
  "ollama": "connected",
  "ollama_url": "http://ollama:11434",
  "documents": 0
}
```

These are expected examples, not recorded container results. `ollama: connected`
confirms that the Ollama API is reachable. Generation also requires a downloaded
model.

Store data and record the count:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/ingest -TimeoutSec 600
$Before = (Invoke-RestMethod http://localhost:8000/stats).documents
$Before
```

The supplied sample produces **4 stored chunks** on a fresh database. The
`documents` field counts chunks, and `source_files` counts source files. Repeating
`/ingest` updates the same chunk IDs rather than duplicating the document.

Test persistence in the same PowerShell window:

```powershell
docker compose down
docker compose up -d --wait --wait-timeout 180
$After = (Invoke-RestMethod http://localhost:8000/stats).documents
$After
if ($Before -gt 0 -and $Before -eq $After) { "PASS: data survived" } else { throw "Persistence check failed" }
```

Do not run `/ingest` between restarting and checking `$After`; that would hide a
persistence failure. Use `docker compose down` without `-v`, because `-v` removes
the named volumes and their stored data.

## Ask a grounded question

With the containers running and documents ingested:

```powershell
docker compose exec ollama ollama pull llama3.2:1b
$Body = @{ question = "What is Docker Compose?" } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8000/ask -ContentType "application/json" -Body $Body -TimeoutSec 240
```

Use the model name from `.env` if you changed it. The answer includes `sources`,
`confidence`, and `chunks_retrieved`. Confidence comes from retrieval distance;
it is a heuristic, not a guarantee that the generated answer is correct.

## API behavior

| Endpoint | Behavior |
| --- | --- |
| `GET /` | Returns the running message and configured generation model |
| `GET /health` | Checks ChromaDB and Ollama; returns HTTP 503 when either is unavailable |
| `GET /stats` | Returns stored chunk count, source file count, and model/configuration information |
| `POST /ingest` | Loads `.txt` and `.md` files from `docs/`, chunks by paragraphs, and upserts them into ChromaDB |
| `POST /ask` | Retrieves up to three chunks, applies the distance threshold, and requests a cited answer from Ollama |

Blank questions return HTTP 422. With no stored documents, `/ask` returns a clear
message telling you to ingest first. Missing models or an unavailable Ollama
service return HTTP 503. CORS is enabled for this local practice API.

Add documents to `backend/docs/` and rebuild with `docker compose up --build -d
--wait` before ingesting them. Documents are copied into the image during the
build. Startup does not automatically ingest documents. Chroma's embedding cache
is in the backend container, so a recreated container may download it again on
the next ingest or search; the stored ChromaDB data remains in the named volume.

## If a command fails

- **Cannot connect to the Docker API / missing dockerDesktopLinuxEngine pipe:**
  open Docker Desktop, wait for the engine, then rerun `docker info`.
- **Port already in use:** change `BACKEND_PORT` or `OLLAMA_PORT` in `.env` to an
  unused host port, such as `8001` or `11435`, then rerun Compose. Keep the backend's
  internal `OLLAMA_URL` as `http://ollama:11434`.
- **Container is unhealthy:** run `docker compose ps`, `docker compose logs backend`,
  and `docker compose logs ollama` to see the specific error.
- **First ingestion fails:** check internet access for the embedding-model download
  and rerun `/ingest`.
- **Model is missing:** run `docker compose exec ollama ollama pull llama3.2:1b`.

Stop the project when finished:

```powershell
docker compose down
```

## Requirement coverage and verification

| Assignment requirement | Implementation / check |
| --- | --- |
| Guided project structure | `backend/` contains Dockerfile, main.py, requirements.txt, and docs/sample.txt; `.env` and Compose file are in the root |
| FastAPI backend built from Dockerfile | `backend.build: ./backend` |
| Ollama image | `ollama.image: ollama/ollama:latest` |
| Both port mappings | Backend 8000 and Ollama 11434 by default |
| Backend knows the Ollama address | `OLLAMA_URL: http://ollama:11434` |
| Named data and model volumes | `chroma_data:/app/chroma_data` and `ollama_data:/root/.ollama` |
| Backend starts after Ollama | `depends_on` with `condition: service_healthy` |
| Build/start and root/health checks | `verify.ps1` runs Compose and checks actual endpoint responses |
| Nonzero data survives down/up | The script compares counts and does not ingest after restarting |

The project files implement the assignment. Completion of its runtime checks
requires successful execution on a computer with Docker running. Expected
responses in this README do not substitute for those checks.

Local validation completed on October 8, 2026, using Python 3.12: dependencies
resolved without conflicts; the Compose YAML and required file structure were
checked; root, health, statistics, ingestion, question validation, Swagger, and
CORS behavior passed API checks. Real Chroma embeddings and retrieval were used.
Repeated ingestion preserved four chunks, and a separate Python process read the
same four stored chunks without ingesting again. Missing-model, upstream-error,
and invalid-response cases were checked with an Ollama HTTP test fixture.
Docker image builds, Compose execution, and real Ollama generation were not run
in that environment because Docker was unavailable. The Dockerfile's Python 3.11
runtime must still be verified through the included container test.
The PowerShell script parsed without syntax errors; execution on Windows remains
part of the container verification step.

Official references:

- [Docker Compose startup and health dependencies](https://docs.docker.com/compose/how-tos/startup-order/)
- [Docker Compose networking](https://docs.docker.com/compose/how-tos/networking/)
- [Docker Compose down and volume removal](https://docs.docker.com/reference/cli/docker/compose/down/)
- [Ollama Docker setup](https://docs.ollama.com/docker)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Chroma embedding functions](https://docs.trychroma.com/docs/embeddings/embedding-functions)
