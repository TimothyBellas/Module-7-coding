"""Load course notes, chunk them, and rebuild a persistent ChromaDB index."""

import argparse
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_PATH = ROOT / "chroma_data"
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once per Python process."""
    return SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed character windows with a shared overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive; overlap must be in [0, chunk_size).")

    text = text.strip()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """Read nonempty UTF-8 .txt and .md files, sorted by relative filename."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Document folder does not exist: {docs_dir}")
    documents = []
    for path in sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in {".txt", ".md"}):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({"filename": path.relative_to(docs_dir).as_posix(), "text": content})
    return documents


def get_collection(chroma_path: Path = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
    """Create or retrieve a persistent collection using cosine distance."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )


def ingest(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    docs_dir: Path = DOCS_DIR,
    chroma_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Embed all documents and rebuild this project's Chroma collection.

    Chunk IDs, source filenames, zero-based chunk indexes, chunk size and overlap
    are saved as metadata. Re-indexing deletes old chunks, including removed files.
    Other collections in the same Chroma directory are untouched. The model is
    encoded before replacement, but an interrupted database write may require
    re-indexing to finish the new collection.
    """
    chunk_text("", chunk_size, overlap)  # Validate before reading or replacing data.
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No nonempty .txt or .md documents found in {docs_dir}")

    records = []
    for doc in documents:
        for index, content in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            records.append((
                f"{doc['filename']}::chunk-{index}",
                content,
                {"source": doc["filename"], "chunk_index": index,
                 "chunk_size": chunk_size, "overlap": overlap},
            ))

    # Finish embeddings before replacing the existing index. A model failure
    # cannot erase a previously working collection.
    model = get_model()
    embeddings = []
    for start in range(0, len(records), 64):
        batch = [item[1] for item in records[start:start + 64]]
        embeddings.extend(model.encode(batch, normalize_embeddings=True).tolist())

    client = chromadb.PersistentClient(path=str(chroma_path))
    if collection_name in [getattr(item, "name", item) for item in client.list_collections()]:
        client.delete_collection(name=collection_name)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    batch_size = min(64, client.get_max_batch_size())
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        collection.upsert(
            ids=[item[0] for item in batch],
            documents=[item[1] for item in batch],
            metadatas=[item[2] for item in batch],
            embeddings=embeddings[start:start + batch_size],
        )
    summary = {"documents": len(documents), "chunks": len(records),
               "chunk_size": chunk_size, "overlap": overlap}
    print(f"Indexed {summary['documents']} documents into {summary['chunks']} chunks "
          f"(size {chunk_size}, overlap {overlap}).")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into persistent ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)
