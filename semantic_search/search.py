"""Query the indexed course notes and return ranked chunks with cosine scores."""

import chromadb

from ingest import CHROMA_PATH, COLLECTION_NAME, get_model


def get_collection():
    """Return the current persisted collection, or None before first indexing."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if COLLECTION_NAME not in [getattr(item, "name", item) for item in client.list_collections()]:
        return None
    return client.get_collection(name=COLLECTION_NAME)


def search(
    query: str,
    n_results: int = 5,
    sources: list[str] | None = None,
    distance_threshold: float | None = None,
) -> list[dict]:
    """Return up to N matches, best first, optionally filtering source and distance.

    Scores are cosine similarity (1 - cosine distance), not probabilities.
    An empty query, zero results requested, missing index, or empty source
    selection produces an empty list. Passing None for sources searches all.
    """
    if not query or not query.strip() or n_results <= 0 or sources == []:
        return []
    if distance_threshold is not None and not 0 <= distance_threshold <= 2:
        raise ValueError("distance_threshold must be between 0 and 2 (cosine distance).")
    collection = get_collection()
    if collection is None or collection.count() == 0:
        return []

    where = {"source": {"$in": sources}} if sources is not None else None
    available = len(collection.get(where=where, include=[])["ids"]) if where else collection.count()
    if available == 0:
        return []

    query_vector = get_model().encode([query.strip()], normalize_embeddings=True)[0].tolist()
    response = collection.query(
        query_embeddings=[query_vector], n_results=min(n_results, available),
        where=where, include=["documents", "metadatas", "distances"],
    )
    results = []
    for content, metadata, distance in zip(
        response["documents"][0], response["metadatas"][0], response["distances"][0]
    ):
        if distance_threshold is not None and distance > distance_threshold:
            continue
        results.append({
            "text": content, "source": metadata["source"],
            "chunk_index": int(metadata["chunk_index"]),
            "distance": float(distance), "score": 1.0 - float(distance),
        })
    return results


def get_collection_stats() -> dict:
    """Return total chunks, unique source files and the active chunk settings."""
    empty = {"total_chunks": 0, "unique_sources": 0, "source_names": [],
             "chunk_size": None, "overlap": None}
    collection = get_collection()
    if collection is None or collection.count() == 0:
        return empty
    metadata = collection.get(include=["metadatas"])["metadatas"]
    names = sorted({entry["source"] for entry in metadata})
    return {"total_chunks": collection.count(), "unique_sources": len(names),
            "source_names": names, "chunk_size": metadata[0].get("chunk_size"),
            "overlap": metadata[0].get("overlap")}
