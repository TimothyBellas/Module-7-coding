"""Evaluate source-level retrieval and repeat the same queries at two sizes."""

import argparse
import json
from pathlib import Path

from ingest import ROOT, ingest
from search import search


# Source-level relevance labels were chosen before comparing the two indices.
EVAL_SET = [
    {"query": "How do Python virtual environments isolate project dependencies?",
     "relevant_sources": ["python-fundamentals.txt"]},
    {"query": "How do API endpoints validate request data and report HTTP errors?",
     "relevant_sources": ["fastapi.txt", "rest-apis.txt"]},
    {"query": "How do SQLAlchemy database tools and FastAPI dependencies work together?",
     "relevant_sources": ["sql-databases.txt", "fastapi.txt"]},
    {"query": "How does Streamlit rerun the script and retain values after widgets change?",
     "relevant_sources": ["streamlit.txt"]},
    {"query": "How can vector retrieval help an LLM avoid hallucinations?",
     "relevant_sources": ["embeddings-and-vectors.txt", "llms-and-ai.txt"]},
]


def precision_recall(
    retrieved_sources: list[str], relevant_sources: list[str]
) -> tuple[float, float]:
    """Compute unique-source precision and recall without counting duplicate chunks."""
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)
    correct = len(retrieved & relevant)
    return correct / len(retrieved) if retrieved else 0.0, correct / len(relevant) if relevant else 0.0


def evaluate(n_results: int = 3, distance_threshold: float | None = None) -> dict:
    """Run the five fixed queries and return per-query top results and mean scores."""
    if n_results <= 0:
        raise ValueError("n_results must be positive")
    cases = []
    for item in EVAL_SET:
        hits = search(item["query"], n_results=n_results, distance_threshold=distance_threshold)
        precision, recall = precision_recall(
            [hit["source"] for hit in hits], item["relevant_sources"]
        )
        cases.append({
            "query": item["query"], "relevant_sources": item["relevant_sources"],
            "precision": round(precision, 4), "recall": round(recall, 4),
            "top_results": [{"source": hit["source"], "chunk_index": hit["chunk_index"],
                             "score": round(hit["score"], 4),
                             "relevant": hit["source"] in item["relevant_sources"],
                             "text_preview": hit["text"][:120]}
                            for hit in hits],
        })
    return {"n_results": n_results, "distance_threshold": distance_threshold,
            "queries": cases,
            "average_precision": round(sum(c["precision"] for c in cases) / len(cases), 4),
            "average_recall": round(sum(c["recall"] for c in cases) / len(cases), 4)}


def print_report(report: dict) -> None:
    """Print each query's ranked sources, score, relevance and aggregate metrics."""
    for case in report["queries"]:
        print(f"\nQuery: {case['query']}")
        for rank, hit in enumerate(case["top_results"], 1):
            label = "relevant" if hit["relevant"] else "not relevant"
            print(f"  {rank}. {hit['source']} [chunk {hit['chunk_index']}] "
                  f"score={hit['score']:.3f} ({label})")
        if not case["top_results"]:
            print("  No matching chunks.")
        print(f"  Precision: {case['precision']:.3f} | Recall: {case['recall']:.3f}")
    print(f"\nAverage precision: {report['average_precision']:.3f} | "
          f"Average recall: {report['average_recall']:.3f}")


def run_experiment(sizes: tuple[int, ...] = (200, 500), overlap: int = 50) -> dict:
    """Re-index at each size, run the same queries, and save detailed JSON results."""
    reports = []
    for size in sizes:
        print(f"\n=== Chunk size {size}; overlap {overlap} ===")
        summary = ingest(chunk_size=size, overlap=overlap)
        report = evaluate(n_results=3)
        print_report(report)
        reports.append({"index": summary, "evaluation": report})
    output = {"model": "all-MiniLM-L6-v2", "metric": "cosine",
              "unit": "unique source files in the top 3 chunks", "runs": reports}
    destination = ROOT / "experiment_results.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {destination.name}. The final index uses size {sizes[-1]}.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the indexed documents")
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Maximum cosine distance; lower is stricter")
    parser.add_argument("--experiment", action="store_true",
                        help="Re-index at 200 and 500 chars with fixed overlap 50")
    args = parser.parse_args()
    if args.experiment:
        run_experiment()
    else:
        print_report(evaluate(args.n_results, args.threshold))
