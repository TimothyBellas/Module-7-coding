# My RAG pipeline with reliability guardrails

This extends the Lesson 3 project. It loads six course-topic text files, chunks
them by paragraph, retrieves the top three candidates from persistent ChromaDB,
filters weak matches, and asks local Ollama to answer from accepted context.
The public `answer_question()` function returns a Python dictionary.

## Four guardrails

| Guardrail | Implementation |
| --- | --- |
| Distance threshold | `filter_chunks()` retains only distances strictly below a configurable threshold; the default is `1.0`. If nothing passes, return a no-relevant-information response without calling Ollama. |
| Confidence | `confidence_level()` uses the best accepted distance: below `0.5` is `high`, below `1.0` is `medium`, and `1.0` or above is `low`. No accepted chunks also means `low`. |
| Stronger prompt | `SYSTEM_PROMPT` says never make up information, say "I don't know" when unsure, always cite sources for factual claims, answer only supported parts, and clarify ambiguous questions. |
| Structured response | Every response contains `answer`, `sources`, `confidence`, and `chunks_retrieved`, including no-match refusals and execution errors. |

`sources` lists the exact labels of the accepted retrieved chunks. It is a
retrieval trace, including when the answer is a clarification. An answer's
inline citations identify which chunks support its factual claims.
`chunks_retrieved` counts accepted chunks, from zero to three; the console
shows all raw candidates and whether each was accepted or filtered out.

Confidence describes retrieval similarity, not a probability of answer
correctness. A close match may still omit part of a question. A generation
error retains the retrieval confidence; the error message and test report
explicitly identify the failed generation.

At the default threshold of `1.0`, any accepted chunk has a distance below
`1.0`, so a generated answer's confidence will be `high` or `medium`.
The `low` generation case can occur with a threshold above `1.0`.

An additional question-only clarity check runs after filtering and before
context-based generation. It asks for a clarification when the subject is
undefined, so retrieved text cannot supply an assumed meaning for "it".
Ollama returns this internal decision using a JSON schema; Python validates it.
An explicitly named document topic or code identifier can continue directly
to the RAG prompt. Other questions use the clarity check first and then generate
only if clear. A no-match query still makes zero Ollama calls.

The default model is `qwen2.5:7b`, used for this version's final live test.
You can select another installed model with `--model`; review its answers
and citations by running the same four-query test.

## Windows PowerShell setup

Extract the ZIP. Open its `my-rag` folder in VS Code and open PowerShell there.
The pinned ChromaDB version was checked with Python 3.12; using Python 3.12
avoids compatibility differences with very new Python versions.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b
.\.venv\Scripts\python.exe my_rag.py
```

Use `py -m venv .venv` if your available compatible Python is already the
default. These commands call the virtual environment's Python directly,
so activation is optional. Ollama is a separate installation; installing
`requirements.txt` does not install its server or generation model.

Open the Ollama app. If it is not running, start it in a second PowerShell
window and leave that window open:

```powershell
ollama serve
```

Check installed models with `ollama list`. If `ollama` is not recognized,
install [Ollama for Windows](https://ollama.com/download/windows) and reopen
PowerShell. Both MiniLM's first download and the Ollama model pull require
internet; later runs can use their cached models locally.

## Run the four required questions

```powershell
.\.venv\Scripts\python.exe my_rag.py --test
Get-Content .\test_results.md
```

This command uses the live pipeline for every question, prints each response
dictionary as readable JSON, and replaces `test_results.md` with actual output,
retrieved distances, filter decisions, and expected behavior. It exits with
code `1` if retrieval or generation fails. A no-match refusal is a successful
execution. Review model answers against the documents; completion alone does
not verify answer quality.

| Query type | Question | Expected behavior |
| --- | --- | --- |
| In-scope | What is st.session_state, and why is it useful in Streamlit? | Explain session values across reruns and cite the Streamlit document. |
| Partially in-scope | What is FastAPI, and how do I deploy it to AWS Lambda? | Explain FastAPI with a citation; say "I don't know" about deployment steps missing from these notes. |
| Out-of-scope | What is the best recipe for chocolate brownies? | Decline to invent a recipe; return a filter refusal if no candidates pass, otherwise the model should admit the context lacks a recipe. |
| Ambiguous | How do I make it remember things? | Ask what "it" refers to and what needs remembering. If no candidates pass, the no-information guardrail takes precedence. |

The project keeps Lesson 3's cosine distance metric. A default threshold of
`1.0` is permissive: an unrelated question can still retrieve chunks below it.
The stronger prompt must still decline unsupported answers. To try stricter
filtering on the same four queries:

```powershell
.\.venv\Scripts\python.exe my_rag.py --test --distance-threshold 0.7
```

Try a value and inspect the actual distances rather than assuming one
threshold is appropriate for every corpus or embedding model. Chunks exactly
at the configured threshold are excluded.

## Interactive mode and configuration

```powershell
.\.venv\Scripts\python.exe my_rag.py
.\.venv\Scripts\python.exe my_rag.py --distance-threshold 0.7
.\.venv\Scripts\python.exe my_rag.py --model qwen2.5:7b
```

Ask questions until you type `quit`. A failed Ollama connection returns an
actionable error in the same dictionary format and keeps the loop running.
Use `--ollama-url` if your server runs at another address.

Example response shape (illustrative, not an observed answer):

```json
{
  "answer": "I don't know. No relevant information was found in the documents.",
  "sources": [],
  "confidence": "low",
  "chunks_retrieved": 0
}
```

## Verification files

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_guardrails.py
```

`test_guardrails.py` uses controlled fixtures to check exact boundaries,
excluded-context behavior, generation skipping, response shape, failures,
the question-only clarity check, and the four-query report. It uses Python's built-in unittest, so no extra
testing package is required. These controlled checks do not evaluate real
LLM answers. `verification_results.md` records checks actually performed
for this version and any live-test limitations.

`lesson3_test_results.md` and `lesson3_verification_results.md` preserve the
earlier version's results. They are historical evidence, not verification
of this enhanced version. `test_results.md` is the current four-query run.

## Project files

| File or folder | Purpose |
| --- | --- |
| `my_rag.py` | Complete ingestion, retrieval, guardrails, Ollama generation, interactive loop, and four-query runner |
| `docs/` | Six UTF-8 course-topic files; 24 paragraph chunks |
| `chroma_data/` | Existing persistent collection, synchronized on startup |
| `requirements.txt` | ChromaDB and requests dependencies |
| `test_guardrails.py` | Controlled implementation checks |
| `test_results.md` | Current actual four-query output |
| `verification_results.md` | Current verification results and limitations |
| `threshold_test_results.json` | Actual no-match responses at a stricter threshold of 0.7 |
| `lesson3_*.md` | Preserved reports from the previous pipeline |

Editing or removing a document updates or removes its chunks when the script
starts. Unchanged paragraphs are not re-embedded. Keep paragraphs short enough
for MiniLM; very long paragraphs can exceed the model's input limit.

These guardrails implement the lesson's reliability requirements. Prompt
instructions and retrieval confidence do not guarantee factual correctness;
review answers and citations when changing models, documents, or thresholds.

## Official API references

- [Chroma query result structure](https://docs.trychroma.com/docs/querying-collections/query-and-get)
- [Chroma collection configuration and distance metrics](https://docs.trychroma.com/docs/collections/configure)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
