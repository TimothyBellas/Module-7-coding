from __future__ import annotations

from collections import Counter
from math import log, sqrt
import re

import chromadb


# Each document has a stable ID, a topic, and text to search. The four topics
# come from Python/API, database, frontend, and AI/search coursework.
DOCUMENTS = {
    "api_basics": ("python_api", "FastAPI creates REST endpoints and validates request bodies with Pydantic."),
    "api_auth": ("python_api", "JWT access tokens authenticate users on protected FastAPI routes."),
    "api_tests": ("python_api", "Pytest checks API responses using TestClient and a temporary SQLite database."),
    "api_async": ("python_api", "Async FastAPI endpoints handle concurrent requests while waiting for I/O."),
    "db_joins": ("database", "SQL joins combine matching rows from relational database tables."),
    "db_orm": ("database", "SQLAlchemy models map Python classes to relational database tables."),
    "db_queries": ("database", "SQLite queries filter, sort, and aggregate student records."),
    "db_schema": ("database", "A database schema defines tables, fields, primary keys, and foreign keys."),
    "web_html": ("frontend", "HTML structures a webpage with headings, forms, and semantic elements."),
    "web_css": ("frontend", "CSS styles layouts, spacing, and responsive mobile screens."),
    "web_fetch": ("frontend", "JavaScript fetch loads API data into a webpage and updates the DOM without reloading."),
    "web_state": ("frontend", "Streamlit widgets and session state preserve values across dashboard reruns."),
    "ai_vectors": ("ai_search", "Text embeddings represent meaning as vectors for semantic search."),
    "ai_chroma": ("ai_search", "ChromaDB stores document embeddings and finds nearest neighbors."),
    "ai_rag": ("ai_search", "Retrieval augmented generation finds relevant passages before a chatbot answers."),
    "ai_threshold": ("ai_search", "Similarity thresholds remove low-scoring search results before answering."),
}


# Human-labeled relevance: more than one answer can be correct for a query.
# These labels do not participate in indexing or ranking.
EVAL_SET = [
    ("How do I secure API routes with bearer tokens?", ["api_auth"]),
    ("How do I test endpoints with a temporary database?", ["api_tests"]),
    ("How can I combine tables and query student records?", ["db_joins", "db_queries"]),
    ("How can I make a webpage fit a phone screen?", ["web_css"]),
    ("How can a webpage load API data without refreshing?", ["web_fetch"]),
    (
        "How does a chatbot find relevant passages using meaning?",
        ["ai_rag", "ai_vectors", "ai_chroma"],
    ),
    ("How do I preserve dashboard values between interactions?", ["web_state"]),
    ("How can I show database query results on a webpage?", ["db_queries", "web_fetch"]),
    ("Which endpoint blocks unauthorized visitors?", ["api_auth"]),
]


# One coordinate per concept. Related words share a coordinate so a query can
# match a document even when they use different words. This is a teaching-sized
# embedding, not a pretrained language model.
CONCEPT_WORDS = {
    "api": {"api", "fastapi", "endpoint", "endpoints", "route", "routes", "rest", "request", "requests"},
    "auth": {"jwt", "token", "tokens", "bearer", "authenticate", "authentication", "login", "secure", "protected"},
    "tests": {"pytest", "test", "tests", "testing", "testclient", "check", "checks", "temporary"},
    "async": {"async", "concurrent", "waiting", "asynchronous"},
    "database": {"database", "sqlite", "sql", "relational", "table", "tables", "rows", "records"},
    "joins": {"join", "joins", "combine", "connect", "matching"},
    "queries": {"query", "queries", "filter", "sort", "aggregate", "student", "students"},
    "orm": {"sqlalchemy", "orm", "models", "classes", "map"},
    "schema": {"schema", "fields", "primary", "foreign", "keys"},
    "web": {"html", "webpage", "website", "page", "web", "frontend"},
    "styles": {"css", "style", "styles", "layout", "layouts", "spacing", "responsive", "mobile", "phone", "screen", "fit"},
    "fetch": {"javascript", "fetch", "load", "loads", "reloading", "refreshing", "dom", "data", "show"},
    "state": {"streamlit", "dashboard", "widgets", "session", "state", "values", "interactions", "reruns", "preserve"},
    "vectors": {"embedding", "embeddings", "vector", "vectors", "meaning", "semantic"},
    "search": {"search", "find", "finds", "retrieval", "relevant", "results", "document", "documents", "passages", "neighbors"},
    "chatbot": {"chatbot", "generation", "rag", "answers", "answering"},
    "threshold": {"threshold", "thresholds", "similarity", "scoring", "low"},
}

CONCEPTS = tuple(CONCEPT_WORDS)


def concepts_in(text: str) -> set[str]:
    """Convert the words in text to recognized semantic concepts."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {concept for concept, synonyms in CONCEPT_WORDS.items() if words & synonyms}


def build_idf() -> dict[str, float]:
    """Give uncommon concepts more weight across the document collection."""
    counts = Counter(concept for _, text in DOCUMENTS.values() for concept in concepts_in(text))
    return {concept: log((len(DOCUMENTS) + 1) / (counts[concept] + 1)) + 1 for concept in CONCEPTS}


def embed(text: str, idf: dict[str, float]) -> list[float]:
    """Produce a unit-length vector so Chroma's default L2 ranks by overlap."""
    found = concepts_in(text)
    if not found:
        raise ValueError(f"No known concepts in text: {text!r}")
    values = [idf[concept] if concept in found else 0.0 for concept in CONCEPTS]
    norm = sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def make_collection(idf: dict[str, float]):
    """Build a fresh in-memory Chroma collection for repeatable evaluations."""
    client = chromadb.Client()
    collection = client.create_collection(name="course_search_evaluation")
    ids = list(DOCUMENTS)
    collection.add(
        ids=ids,
        documents=[DOCUMENTS[doc_id][1] for doc_id in ids],
        metadatas=[{"topic": DOCUMENTS[doc_id][0]} for doc_id in ids],
        embeddings=[embed(DOCUMENTS[doc_id][1], idf) for doc_id in ids],
    )
    return collection


def evaluate(collection, idf: dict[str, float], n_results: int) -> dict:
    """Report per-query and macro-average precision/recall at a given top-k."""
    print(f"\n=== Evaluation at n_results={n_results} ===")
    per_query = []

    for number, (query, expected_ids) in enumerate(EVAL_SET, start=1):
        result = collection.query(
            query_embeddings=[embed(query, idf)],
            n_results=n_results,
            include=["distances"],
        )
        retrieved_ids = result["ids"][0]
        expected = set(expected_ids)
        hits = expected.intersection(retrieved_ids)
        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(hits) / len(expected) if expected else 0.0
        missed = sorted(expected - hits)
        per_query.append({"query": query, "precision": precision, "recall": recall, "missed": missed, "retrieved": retrieved_ids})
        print(f"Query {number}: P={precision:.1%} R={recall:.1%} | {query}")
        print(f"  Expected: {expected_ids} | Retrieved: {retrieved_ids} | Missed: {missed}")

    avg_precision = sum(item["precision"] for item in per_query) / len(per_query)
    avg_recall = sum(item["recall"] for item in per_query) / len(per_query)
    print(f"AVERAGE: P={avg_precision:.1%} R={avg_recall:.1%}")
    return {"n_results": n_results, "precision": avg_precision, "recall": avg_recall, "queries": per_query}


def analyze(all_results: list[dict]) -> None:
    """Describe actual strengths, misses, and a concrete next improvement."""
    smallest, middle, largest = all_results
    print("\n=== ANALYSIS ===")
    strong = [
        str(i) for i, item in enumerate(smallest["queries"], 1)
        if item["precision"] == 1.0 and item["recall"] == 1.0
    ]
    weak = [str(i) for i, item in enumerate(smallest["queries"], 1) if item["recall"] < 1.0]
    print(f"Fully answered at top-{smallest['n_results']}: {', '.join(strong) or 'none'}.")
    print(f"Missed some relevant documents at top-{smallest['n_results']}: {', '.join(weak) or 'none'}.")
    for number, item in enumerate(smallest["queries"], 1):
        if item["recall"] == 0.0:
            print(f"Query {number} failed at top-1: returned {item['retrieved']} instead of {item['missed']}.")
    print(
        f"Average recall changed from {smallest['recall']:.1%} to {middle['recall']:.1%} "
        f"to {largest['recall']:.1%}, while average precision changed from "
        f"{smallest['precision']:.1%} to {middle['precision']:.1%} to {largest['precision']:.1%}."
    )
    remaining = [(i, item["missed"]) for i, item in enumerate(largest["queries"], 1) if item["missed"]]
    if remaining:
        for number, missed in remaining:
            print(f"Query {number} still failed to find {missed} at top-{largest['n_results']}.")
        print("Next: rewrite or split broad documents, add useful synonyms, then rerun the same labeled queries.")
    else:
        print("Every labeled document appeared by the largest setting; extra results lowered precision.")
        print("Next: try a similarity cutoff or reranking, and test harder paraphrases with the same labels.")
    if smallest["queries"][-1]["recall"] == 0.0:
        print("Query 9 exposes a synonym gap: add 'unauthorized' and 'visitors' to the auth concept, then reevaluate.")
    print("The curated concept vocabulary is limited; compare it with a pretrained text embedding model next.")


def main() -> None:
    known_ids = set(DOCUMENTS)
    for query, expected_ids in EVAL_SET:
        if not expected_ids or not set(expected_ids) <= known_ids:
            raise ValueError(f"Invalid relevance labels for query: {query!r}")
    idf = build_idf()
    collection = make_collection(idf)
    print(f"Indexed {collection.count()} documents across 4 topics; {len(EVAL_SET)} labeled queries.")
    results = [evaluate(collection, idf, n_results=n) for n in (1, 3, 5)]
    analyze(results)


if __name__ == "__main__":
    main()
