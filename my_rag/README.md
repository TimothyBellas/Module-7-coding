# My RAG pipeline

This project searches six course-topic documents and asks a local Ollama model
to answer using the three retrieved paragraphs. It prints the retrieved text
and source labels before the answer.

## Files

| File or folder | Purpose |
| --- | --- |
| `my_rag.py` | Ingestion, retrieval, prompt assembly, Ollama generation, interactive loop, and three-question test runner |
| `docs/` | Six UTF-8 text files about Python, FastAPI, SQLAlchemy, Streamlit, embeddings, and RAG |
| `chroma_data/` | Persistent ChromaDB database created and updated by the script |
| `requirements.txt` | Python dependencies |
| `test_results.md` | Actual output of the three assignment questions; `--test` replaces this report |
| `verification_results.md` | Additional implementation checks performed before packaging |

## Setup in Windows PowerShell

Extract the ZIP and open its `my-rag` folder in VS Code. Open a PowerShell
terminal in that folder. Use Python 3.11 or newer (the project was tested with
Python 3.12). Ollama must be installed separately from the Python packages.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull llama3.2:3b
.\.venv\Scripts\python.exe my_rag.py
```

These commands use the virtual environment's Python directly, so activating
the environment is optional. If `py` is not recognized, install Python with
the Windows Python launcher. If `ollama` is not recognized, install
[Ollama for Windows](https://ollama.com/download/windows) and reopen PowerShell.

Open the Ollama app before running the script. If it is not serving requests,
start it in a **second PowerShell window** and leave that window open:

```powershell
ollama serve
```

If `ollama serve` says the port is already in use, Ollama may already be
running. Check it with `ollama list`, then try the RAG script.

Chroma downloads its MiniLM embedding model the first time it needs to embed
text. This requires an internet connection. Ollama's first model pull also needs internet.
After both models have been downloaded, retrieval and generation run locally.
The first answer may take longer while Ollama loads the model.

## Run the three required questions

```powershell
.\.venv\Scripts\python.exe my_rag.py --test
Get-Content .\test_results.md
```

The test command runs these questions through the same pipeline as interactive
mode and records the real retrieved paragraphs, distances, answers or errors,
and expected behavior:

| Query type | Question | Expected behavior |
| --- | --- | --- |
| Answerable | What is st.session_state, and why is it useful in Streamlit? | Explain saved session values across reruns and cite the Streamlit document. |
| Related, not directly documented | How do I deploy a FastAPI application to AWS Lambda? | State that the documents do not provide AWS Lambda deployment instructions. Cite any supported background facts. |
| Outside scope | What is the best recipe for chocolate brownies? | State that the documents do not contain enough information. Do not invent a recipe. |

A successful HTTP response is not proof that the answer is accurate. Review
the actual answers and citations against the expected behavior before submitting.
If Ollama is unavailable or a model is missing, the report records a generation
error rather than presenting it as an LLM answer. Fix the error and rerun `--test`.

## Interactive use

```powershell
.\.venv\Scripts\python.exe my_rag.py
```

Enter a question at `Your question:`. You will see the top three paragraphs
with labels such as `[streamlit_basics.txt:paragraph-2]`, then the generated
answer. Type `quit` to stop. A connection failure displays a message and keeps
the loop running, so you can start Ollama and retry without restarting the script.

To use a different model you already have installed:

```powershell
ollama list
.\.venv\Scripts\python.exe my_rag.py --model llama3.2
.\.venv\Scripts\python.exe my_rag.py --test --model llama3.2
```

Replace `llama3.2` with the exact installed model name if necessary.

## How the code works

1. **Ingest:** `load_chunks()` reads `.txt` files and splits on blank lines.
   `ingest_documents()` saves each paragraph's text, embedding, filename,
   paragraph number, and unique ID in `chroma_data/`. Content hashes avoid
   re-embedding unchanged paragraphs; upserts update changed ones. Obsolete
   chunks are removed when documents are shortened or removed.
2. **Retrieve:** `retrieve_chunks()` embeds the current question with the same
   MiniLM model and queries Chroma for the top three chunks using cosine distance.
3. **Generate:** `build_rag_prompt()` formats the retrieved paragraphs with
   citation labels. `generate_answer()` sends that prompt and a separate system
   message to Ollama's `/api/chat` endpoint and returns its answer.
4. **Repeat:** `interactive_loop()` accepts questions until `quit`.

The system prompt requires context-only answers, source citations, and an
explicit acknowledgment of missing information. This is a prompt instruction,
not a guarantee: small language models can still produce unsupported claims.
The script deliberately retrieves three nearest chunks for every question;
nearest does not necessarily mean relevant or sufficient to answer.

Keep paragraphs reasonably short. A long paragraph can exceed the embedding
model's input limit, so paragraph chunking is best suited to these concise notes.
Edit the six included files or add your own `.txt` files, then restart the script
to synchronize the database.

## Requirement checklist

- Six topic documents in `docs/`.
- Paragraph-based ingestion into persistent ChromaDB.
- Top-three retrieval for each question.
- Context plus system instructions sent to Ollama for generation.
- Source citation instructions and visible retrieved chunks.
- Graceful messages for an unavailable Ollama server, a missing model, and timeouts.
- An interactive loop that ends on `quit`.
- Three built-in query types with saved actual test output.

## API documentation

- [Chroma embedding functions](https://docs.trychroma.com/docs/embeddings/embedding-functions)
- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [Chroma queries and result structure](https://docs.trychroma.com/docs/querying-collections/query-and-get)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama on Windows](https://docs.ollama.com/windows)
