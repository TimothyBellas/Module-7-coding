FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_OLLAMA_URL=http://host.docker.internal:11434 \
    RAG_DOCS_DIR=/app/docs \
    RAG_CHROMA_DIR=/app/chroma_data

WORKDIR /app

# Keep system dependencies in their own reusable layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies BEFORE copying code, so code edits reuse this layer.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY my_rag_api.py ./
COPY docs/ ./docs/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "my_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
