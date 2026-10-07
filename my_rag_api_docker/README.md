# Docker practice: My RAG API

This project packages your uploaded `my_rag_api.py` with a Dockerfile, Python dependencies, and six starter documents. You can replace the files in `docs/` with your own UTF-8 `.txt` or `.md` documents before building.

The API runs in a container. Ollama runs on your Windows computer. The Dockerfile sets `RAG_OLLAMA_URL=http://host.docker.internal:11434` so the API can reach it. You do not need a Windows Python installation or virtual environment to run this image.

## 1. Prepare Docker Desktop and Ollama

Start Docker Desktop and use Linux containers. Open PowerShell and check Docker:

```powershell
docker version
```

The command should show both Client and Server information. If the Server is unavailable, wait for Docker Desktop to finish starting.

Quit Ollama from the Windows system tray first. In a PowerShell window, run these commands and keep that window open:

```powershell
$env:OLLAMA_HOST = "0.0.0.0:11434"
ollama serve
```

This allows Docker Desktop to reach Ollama through the host's network interface. If port 11434 is already in use, quit the existing Ollama application/server before starting this one.

In a second PowerShell window, download both models required by your uploaded API and check the server:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
ollama list
Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
```

If `ollama` is not recognized, reopen PowerShell after installing Ollama. The API needs both the chat model `llama3.2` and the embedding model `nomic-embed-text`.

## 2. Build and run

Extract the ZIP, then open a PowerShell terminal in the `my-rag-api-docker` folder, alongside the Dockerfile. Run:

```powershell
docker build -t my-rag-api .
docker run -p 8000:8000 my-rag-api
```

Keep the container terminal open. When Uvicorn reports that it is running on `0.0.0.0:8000`, open this URL in your browser:

[http://localhost:8000/docs](http://localhost:8000/docs)

## 3. Test in Swagger UI

1. Expand **GET /health**, select **Try it out**, then **Execute**. Expect HTTP **200**, `status: "healthy"`, both dependency checks `ok: true`, and `missing_models: []`.
2. Expand **POST /ingest**, select **Try it out**, replace the request body with `{}`, and select **Execute**. Expect HTTP **200**, `documents_loaded: 6`, and a positive `chunks_ingested` count for the supplied starter documents.
3. Expand **POST /ask**, select **Try it out**, and submit:

```json
{
  "question": "What is retrieval-augmented generation?"
}
```

Expect HTTP **200**, a nonempty `answer`, a nonempty `sources` list, `chunks_retrieved` greater than zero, and `confidence` set to `high`, `medium`, or `low`. Read the answer to confirm it matches the supplied documents and includes numbered citations such as `[1]`. The model's exact wording and retrieved sources can vary.

4. Run **GET /stats**. Expect HTTP **200** with `document_count: 6`, a positive `chunk_count`, `ollama_model: "llama3.2"`, and `embedding_model: "nomic-embed-text"`.
5. Send `{"question": ""}` to **POST /ask**. Expect HTTP **422**.

Save screenshots of the successful ingestion and question response for your assignment. These are expected results to check on your computer, not fabricated test output.

If `/health` reports HTTP 503, inspect `ollama`, `chromadb`, and `missing_models` in the response. A running API can still have an unhealthy dependency. The empty database is not itself a health failure; call `/ingest` before asking questions.

Optional PowerShell checks, in another terminal while the container is running:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/ingest" -ContentType "application/json" -Body '{}' -TimeoutSec 600
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/ask" -ContentType "application/json" -Body '{"question":"What is retrieval-augmented generation?"}' -TimeoutSec 600
Invoke-RestMethod -Uri "http://localhost:8000/stats"
```

If model loading is slow, increase the API's Ollama timeout when starting the container:

```powershell
docker run -p 8000:8000 -e RAG_OLLAMA_TIMEOUT=300 my-rag-api
```

## 4. Verify Docker's dependency-layer cache

Run this from the project folder after the first successful build. The same build context and Docker builder must be used; keep `requirements.txt`, the Dockerfile, and the base image unchanged.

```powershell
docker build --progress=plain -t my-rag-api . 2>&1 | Tee-Object -FilePath build-first.log
Add-Content -Path .\my_rag_api.py -Value "`n# Docker cache practice: changed only a Python comment."
docker build --progress=plain -t my-rag-api . 2>&1 | Tee-Object -FilePath build-rebuild.log
Select-String -Path .\build-rebuild.log -Pattern "pip install", "CACHED", "COPY my_rag_api.py"
```

In `build-rebuild.log`, find the numbered build step containing `RUN python -m pip install --no-cache-dir -r requirements.txt`. The line for that same step should say **CACHED**. The `COPY my_rag_api.py` step should execute again because the Python file changed. This confirms that Docker reused the Python dependency layer.

Do not use `--no-cache` or prune the builder cache for this exercise. The `--no-cache-dir` pip option in the Dockerfile does not disable Docker's layer cache.

## 5. Container management and persistent data

In another PowerShell terminal:

```powershell
docker ps
docker ps -a
docker images
```

To stop your running container, copy its ID from `docker ps` and replace `YOUR_CONTAINER_ID`:

```powershell
docker stop YOUR_CONTAINER_ID
docker rm YOUR_CONTAINER_ID
```

By default, ChromaDB persists in that container's filesystem. Restarting the same container keeps it, but removing the container removes that data. To keep data independently, start a new container with a named volume after stopping the earlier one:

```powershell
docker run -p 8000:8000 -v rag-chroma-data:/app/chroma_data my-rag-api
```

The `.dockerignore` excludes your local `chroma_data` from the build context; it does not delete a Docker volume. When you add or edit documents copied into the image, rebuild, replace the running container with one using the new image, and call `/ingest` again.

For Docker Engine on Linux, supply the host mapping explicitly:

```bash
docker run --add-host=host.docker.internal:host-gateway -p 8000:8000 my-rag-api
```

## Requirement coverage

| Requirement | Included implementation or verification |
| --- | --- |
| Base image `python:3.11-slim` | First line of Dockerfile |
| System dependencies | `build-essential`, with apt lists removed afterward |
| Copy requirements before code | `COPY requirements.txt` precedes package installation and application copies |
| Install packages using `--no-cache-dir` | Dockerfile pip installation command |
| Copy application and `docs/` | Explicit `COPY` instructions for both |
| Expose port 8000 | `EXPOSE 8000` |
| Bind Uvicorn to `0.0.0.0` | JSON-form `CMD` with module `my_rag_api:app` |
| Ignore required local files | `.dockerignore` includes `__pycache__`, `venv`, `.git`, `.env`, and `chroma_data` |
| Build and run | Commands in section 2; verify on your Docker installation |
| Browser `/ingest` and `/ask` tests | Swagger instructions in section 3; verify with real Ollama |
| Cached pip layer after comment edit | Rebuild steps and log inspection in section 4; verify on your Docker builder |

Verification performed in this session is recorded separately in `VALIDATION.md`. A container build, live Ollama answer, and Docker cache result must not be marked complete until they actually run successfully.

## References

- [Docker: optimize cache usage](https://docs.docker.com/build/cache/optimize/)
- [Docker Desktop: connect containers to host services](https://docs.docker.com/desktop/features/networking/networking-how-tos/)
- [Ollama: environment variables and network binding](https://docs.ollama.com/faq)
- [Chroma: configure collections](https://docs.trychroma.com/docs/collections/configure)
