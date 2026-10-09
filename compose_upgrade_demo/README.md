# RAG API configuration refactor

The FastAPI RAG app now reads its configuration through `backend/config.py`.
Its `Settings` class validates environment variables, reads the project-root
`.env` for local runs, and provides defaults when values are missing.
Docker Compose passes `.env` to the backend with `env_file: .env`.

The existing `compose-demo` project name, `documents` collection, embedding
function, and named volumes are preserved.

## Files

| Path | Purpose |
| --- | --- |
| `backend/config.py` | Central Settings class and validation |
| `backend/main.py` | Uses Settings for storage, retrieval, generation, health checks, CORS, and debugging |
| `.env` | Ready-to-run Docker Compose values; no secrets included |
| `.env.example` | Documented values to copy when setting up the project |
| `.gitignore` | Excludes .env and local outputs; keeps .env.example |
| `backend/.dockerignore` | Excludes .env from the active backend build context |
| `docker-compose.yml` | Backend env_file, Ollama service, ports, health checks, and volumes |
| `backend/Dockerfile` | Python 3.11 image; installs requirements before copying code |
| `backend/requirements.txt` | Pinned app dependencies, including pydantic-settings and python-dotenv |
| `verify-config.ps1` | Changes MODEL_NAME, recreates the backend, and records actual API results |
| `verify.ps1` | Previous Compose connectivity and data persistence checks |
| `tests/` | Configuration and API behavior checks without Docker or model downloads |

## Required settings

| Variable | Python default without .env | Included Docker .env value |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | `http://ollama:11434` |
| `MODEL_NAME` | `llama3.2:1b` | `llama3.2:1b` |
| `CHROMA_PATH` | `backend/chroma_data` | `/app/chroma_data` |
| `MAX_RESULTS` | `3` | `3` |
| `CONFIDENCE_THRESHOLD` | `1.0` | `1.0` |
| `DEBUG` | `false` | `false` |

Additional settings cover `DOCS_PATH`, `COLLECTION_NAME`, HTTP timeouts,
generation temperature, ingestion batch size, allowed CORS origins, and Chroma
telemetry. The sample .env documents all of them. `BACKEND_PORT`,
`OLLAMA_PORT`, and `OLLAMA_HOST` configure Compose.

`MAX_RESULTS` limits the number of retrieved chunks. Chunks must have cosine
distance **strictly below** `CONFIDENCE_THRESHOLD` to reach Ollama.
Confidence is high below half that threshold, medium below the threshold, and
low when no chunks pass. At the default 1.0 threshold, the original 0.5/1.0
confidence boundaries are retained. Confidence is a retrieval heuristic.

`DEBUG` controls FastAPI's debug behavior and the app logger's level.
Values such as `true` and `false` are parsed as booleans.
Invalid URLs, blank model names, invalid booleans, and out-of-range numeric
values fail with a validation error at startup.

## Run and verify on Windows

1. Extract the ZIP and open the `compose-demo` folder in VS Code.
2. Open Docker Desktop and wait for the engine to run. Use Linux containers.
3. Open PowerShell in that project folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify-config.ps1
```

This builds and starts the services, records the current model, edits
`MODEL_NAME` in .env to a different value, recreates the backend, and checks
that both `/` and `/stats` report the changed model. It also checks Ollama
connectivity and the stored chunk count. No ingestion occurs during this check.

The script leaves the new model in .env and the containers running.
It saves the actual results to `config-verification-results.txt`.
Keep that transcript as assignment evidence.

To choose the test model explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify-config.ps1 -NewModel "llama3.2:3b"
```

The chosen name must differ from the currently configured model. A model
download is not required to demonstrate the setting change in the API.
Before generating answers with it, pull that model inside Ollama:

```powershell
$Model = (Invoke-RestMethod http://localhost:8000/).model
docker compose exec ollama ollama pull $Model
```

Docker handles the Python environment; no host Python installation or virtual
environment is needed for these Docker commands.

## Manual model-change check

These commands use the included default backend port, 8000.

```powershell
docker info
docker compose up --build -d --wait --wait-timeout 180
(Invoke-RestMethod http://localhost:8000/).model
notepad .env
```

Change `MODEL_NAME=llama3.2:1b` to `MODEL_NAME=llama3.2:3b`, save, then run:

```powershell
docker compose up -d --force-recreate --no-deps --wait --wait-timeout 180 backend
(Invoke-RestMethod http://localhost:8000/).model
Invoke-RestMethod http://localhost:8000/stats
```

The new model should appear as `llama3.2:3b` in both responses.
These are expected values; the script records the results from your containers.

Use **recreation** after changing .env. `docker compose restart` keeps a
container's existing environment variables and does not load changed values.
No image rebuild is needed for a subsequent .env-only change.

## .env management and precedence

For Python Settings, process environment variables override .env values, and
.env values override class defaults. Settings load once per API process.

In this Compose project, the backend has no `environment` entries that
override `env_file`. Compose reads the root .env and injects its values into
the backend. The .env file is excluded from the image; runtime configuration
is supplied when the container is created.

The ZIP includes a non-secret .env so it can run immediately. A Git clone
contains .env.example; create .env if missing:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Keep actual secrets in your ignored .env and document only placeholders in
.env.example. If .env was already tracked by Git, untrack it once with
`git rm --cached -- .env`, then commit the removal and the ignore rules.

The active Docker build context is `backend/`, so its .dockerignore owns the
image exclusions. The root .env also lies outside that build context.

## Data and local settings

Keep `CHROMA_PATH=/app/chroma_data` with the existing Compose volume mount.
If you change this path for Docker, update the volume's container mount path
to match it. Keep `COLLECTION_NAME=documents` to continue using the existing
collection.

`docker compose down` preserves the named volumes. The previous assignment's
full persistence check is still available:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
```

Use `-TestRag` with that script to pull the configured generation model and
test an actual grounded answer. The first ingestion or retrieval may download
Chroma's default all-MiniLM-L6-v2 embedding model.

For local Python execution, override the Docker-only hostname and paths.
From the project folder, with the app dependencies installed in Python 3.11
or 3.12:

```powershell
$env:OLLAMA_URL = "http://localhost:11434"
$env:CHROMA_PATH = "./chroma_data"
$env:DOCS_PATH = "./docs"
python -m uvicorn main:app --app-dir backend --reload
```

Relative storage/document paths resolve under backend/. Ollama must run locally.
Open [Swagger UI](http://localhost:8000/docs) to use /ingest, /ask, /stats,
and /health.

## Verification status

The included tests check defaults, .env types and Windows encoding, environment
precedence, invalid settings, model changes across fresh API processes,
configured Chroma storage, settings in the Ollama request, retrieval filtering,
and existing validation/error behavior.

Run them using Python 3.11 or 3.12:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

On October 9, 2026, all 14 checks passed with Python 3.12 and the pinned project
dependencies. Dependency compatibility, Compose YAML structure, Git ignore
behavior, and both PowerShell scripts' syntax were also checked successfully.

Docker was unavailable in the checking environment. Docker image builds,
Compose execution, and a real Ollama model change must still be verified with
the Windows script. The Ollama test fixture does not generate real model answers.

## Official references

- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Compose env_file](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/)
- [Compose restart environment behavior](https://docs.docker.com/reference/cli/docker/compose/restart/)
- [Compose up and container recreation](https://docs.docker.com/reference/cli/docker/compose/up/)
