# Verification results

Checked at 2026-10-01T21:22:57+00:00 (UTC).

Python 3.12.14; ChromaDB 1.5.9; requests 2.34.2.
Live generation: Ollama 0.35.0, local `qwen2.5:7b`, temperature 0, context length 4096.
The default distance threshold is 1.0, using the existing cosine-distance collection.

## Implementation checks

All **19** checks in `test_guardrails.py` passed. These controlled checks verify
strict threshold boundaries, all three confidence levels, excluded prompt
content, zero generation calls for no-match queries, dictionary shape,
connection and retrieval errors, invalid configuration, question-only clarity
checking, explicit subjects, invalid clarity responses, and the four-query report.
They do not substitute for the separate live answer review below.

Six text files produce 24 unique paragraph chunks. Live startup reopened the
existing persistent collection, retained all 24 chunks, and made zero updates
because the document contents had not changed. Real MiniLM embeddings were
used to query the collection; no simulated retrieval was used in the live run.

## Four live query types

The final `test_results.md` contains the actual candidates, distances, filter
decisions, and response dictionaries for all four required questions. The
command completed with zero retrieval or generation errors. Each response had
exactly the four requested fields. All source labels corresponded to accepted
chunks; generated inline citations referenced real retrieved labels. All four
responses were under 150 whitespace-separated words.

| Query type | Best distance | Confidence | Accepted chunks | Observed behavior | Review |
| --- | --- | --- | --- | --- | --- |
| In-scope | 0.1988 | high | 3 | Defined session state and its use across reruns; cited the supporting Streamlit paragraph. | PASS |
| Partially in-scope | 0.3689 | high | 3 | Defined FastAPI with a citation and said it did not know the undocumented AWS Lambda deployment steps. | PASS |
| Out-of-scope | 0.9201 | medium | 3 | Said it did not know a brownie recipe from the provided context; invented no recipe or unrelated background. | PASS |
| Ambiguous | 0.8134 | medium | 3 | Asked which application or tool the user meant and what it should do; assumed no framework. | PASS |

The supported factual claims were compared with the cited document paragraphs.
The ambiguity check uses a question-only decision when the subject is not
explicitly named; the clarification does not present document facts. Its
`sources` field remains a trace of accepted retrieval candidates.

## Real no-match filtering

With the final code and a threshold of **0.7**, both the out-of-scope and
ambiguous questions had zero accepted chunks. Each returned the no-relevant-
information response, empty sources, low confidence, and `chunks_retrieved: 0`.
A generation spy confirmed `generate_answer()` was never called. Their actual
response dictionaries are saved in `threshold_test_results.json`.

## Scope of these results

Confidence measures the best accepted retrieval distance, not answer accuracy.
At threshold 1.0, the brownie question still retrieves weak chunks below the
cutoff, so the prompt must decline unsupported information. At 0.7, filtering
rejects those candidates before any Ollama request.

These results describe this corpus, code version, and model. Review new answers
and citations when changing documents, model, or threshold. The `lesson3_*.md`
files are preserved historical reports, not verification of this version.
