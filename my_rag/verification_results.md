# Implementation verification

Checked at 2026-09-30T21:15:50+00:00 (UTC).

Python 3.12.14; ChromaDB 1.5.9. Real MiniLM embeddings were used.

These checks verify pipeline behavior. The separate `test_results.md` contains the actual Ollama answers to the three assignment questions.

| Check | Result | Evidence |
| --- | --- | --- |
| Corpus and chunking | PASS | Six text files produce 24 nonempty paragraph chunks. |
| Persistence | PASS | A new Python process reopened the same database and found 24 chunks. |
| Repeat ingestion | PASS | Unchanged documents require zero updates; all 24 IDs remain unique. |
| Semantic retrieval | PASS | All three questions return three chunks; the session-state paragraph ranks first for the answerable question. |
| Prompt assembly | PASS | The prompt contains actual retrieved text, exact source labels, the question, and citation instructions. |
| Unavailable Ollama | PASS | A real refused connection returns an actionable message without an uncaught exception. |
| API error handling | PASS | Controlled HTTP error fixtures verify missing-model, HTTP 500, invalid JSON, and unexpected response-shape messages. These are transport tests, not LLM generations. |
| Interactive loop | PASS | Blank input is handled, a generation failure leaves the loop running, mixed-case quit exits, and retrieved text precedes the error/answer area. |
| Source synchronization | PASS | In a temporary database, editing/shortening one file updates its text and removes old paragraphs; deleting the file removes its remaining chunk. |

## Review of generated answers

Ollama 0.35.0 generated the final answers with `llama3.2:3b` and temperature 0. The complete output is in `test_results.md`.

| Required query type | Observed answer | Review |
| --- | --- | --- |
| Answerable | Defined session state and explained preserved values across reruns; cited `[streamlit_basics.txt:paragraph-2]`. | PASS: claims match the cited paragraph. |
| Related but undocumented | Said that AWS Lambda deployment is not documented and declined to supply instructions. | PASS: no unsupported deployment steps. |
| Outside scope | Said that brownie recipes are not covered and declined to supply one. | PASS: no invented recipe or unrelated citation. |

All three answers were under 150 words. The cited label in the supported answer matches a retrieved chunk. These results describe this run; review new output when rerunning with another model or different documents.
