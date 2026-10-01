# RAG test results

Run time (UTC): 2026-10-01T21:47:29+00:00

Ollama model: qwen2.5:7b

Distance threshold: 1 (strictly below; cosine distance).

Stored paragraph chunks: 24

These are actual outputs from this run. Expected behavior is a review guide, not a generated answer or an automatic quality score.

## Test 1: In-scope

**Question:** What is st.session_state, and why is it useful in Streamlit?

**Expected behavior:** Explain that st.session_state preserves values between reruns within a user's session; cite streamlit_basics.txt.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved candidates (lower cosine distance = closer match):
Only distances < 1 will be sent to Ollama.

1. [streamlit_basics.txt:paragraph-2] | distance=0.1988 | ACCEPTED
st.session_state is Streamlit's container for session values. It stores values across reruns within a user's Streamlit session. It is useful for preserving a quiz score, a current question number, login information, and chat history, so users retain their progress when the script reruns. Initialize a key only if it does not already exist so a rerun does not reset the saved value.

2. [streamlit_basics.txt:paragraph-4] | distance=0.5175 | ACCEPTED
st.cache_data caches the results of data-returning functions, such as loading a dataset or fetching API data. Reusing a cached result can reduce repeated work. A function's clear method or st.cache_data.clear can clear the corresponding cache, but clearing cached data does not automatically reset widget values in st.session_state.

3. [streamlit_basics.txt:paragraph-1] | distance=0.6240 | ACCEPTED
Streamlit runs a Python script from top to bottom when a user interacts with a widget. This rerun model makes it straightforward to rebuild the page, but ordinary local variables do not preserve their values between separate script executions.

Accepted chunks: 3. Checking question and generating response...

Structured response:
{
  "answer": "st.session_state is Streamlit's container for session values, storing values across reruns within a user's Streamlit session. It is useful for preserving a quiz score, a current question number, login information, and chat history, so users retain their progress when the script reruns. [streamlit_basics.txt:paragraph-2]",
  "sources": [
    "[streamlit_basics.txt:paragraph-2]",
    "[streamlit_basics.txt:paragraph-4]",
    "[streamlit_basics.txt:paragraph-1]"
  ],
  "confidence": "high",
  "chunks_retrieved": 3
}
```

## Test 2: Partially in-scope

**Question:** What is FastAPI, and how do I deploy it to AWS Lambda?

**Expected behavior:** Explain FastAPI using a citation, then say I don't know about AWS Lambda deployment because these notes do not cover it. Do not invent steps.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved candidates (lower cosine distance = closer match):
Only distances < 1 will be sent to Ollama.

1. [fastapi_basics.txt:paragraph-1] | distance=0.3689 | ACCEPTED
FastAPI is a Python framework for building web APIs. An endpoint combines a URL path with an HTTP method. For example, GET /tasks can retrieve tasks, while POST /tasks can create a task. CRUD means create, read, update, and delete.

2. [fastapi_basics.txt:paragraph-2] | distance=0.4669 | ACCEPTED
FastAPI uses Pydantic models to validate request data. A request schema describes expected fields and types. Field constraints can set a minimum number or a maximum string length. Invalid request data normally produces an HTTP 422 validation response.

3. [fastapi_basics.txt:paragraph-3] | distance=0.4818 | ACCEPTED
FastAPI generates interactive API documentation at /docs and alternative documentation at /redoc. Run a development application whose file is main.py and whose FastAPI instance is named app with python -m uvicorn main:app --reload.

Accepted chunks: 3. Checking question and generating response...

Structured response:
{
  "answer": "I don't know how to deploy FastAPI to AWS Lambda. FastAPI is a Python framework for building web APIs [fastapi_basics.txt:paragraph-1].",
  "sources": [
    "[fastapi_basics.txt:paragraph-1]",
    "[fastapi_basics.txt:paragraph-2]",
    "[fastapi_basics.txt:paragraph-3]"
  ],
  "confidence": "high",
  "chunks_retrieved": 3
}
```

## Test 3: Out-of-scope

**Question:** What is the best recipe for chocolate brownies?

**Expected behavior:** Say that the provided documents do not contain enough information; do not supply a recipe or cite unrelated course notes.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved candidates (lower cosine distance = closer match):
Only distances < 1 will be sent to Ollama.

1. [sqlalchemy_basics.txt:paragraph-1] | distance=0.9201 | ACCEPTED
SQLAlchemy is a Python library for working with relational databases. Its object-relational mapper maps Python model classes to database tables. An instance of a mapped class represents a row, and mapped attributes represent columns.

2. [sqlalchemy_basics.txt:paragraph-3] | distance=0.9414 | ACCEPTED
A SQLAlchemy session manages work with database records. Add a new model instance to the session and call commit to persist the transaction. Refresh can reload database-generated values. Rollback cancels pending changes after a failed transaction.

3. [embeddings_and_search.txt:paragraph-4] | distance=0.9523 | ACCEPTED
ChromaDB stores embeddings together with text, unique IDs, and metadata. PersistentClient saves the database to a specified folder so its data survives process restarts. A collection query returns nearest chunks, and source metadata helps show where each retrieved chunk came from.

Accepted chunks: 3. Checking question and generating response...

Structured response:
{
  "answer": "I don't know the best recipe for chocolate brownies based on the provided context.",
  "sources": [
    "[sqlalchemy_basics.txt:paragraph-1]",
    "[sqlalchemy_basics.txt:paragraph-3]",
    "[embeddings_and_search.txt:paragraph-4]"
  ],
  "confidence": "medium",
  "chunks_retrieved": 3
}
```

## Test 4: Ambiguous

**Question:** How do I make it remember things?

**Expected behavior:** Ask what 'it' refers to and what should be remembered; do not assume the user means Streamlit, ChromaDB, or a database session.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved candidates (lower cosine distance = closer match):
Only distances < 1 will be sent to Ollama.

1. [streamlit_basics.txt:paragraph-2] | distance=0.8134 | ACCEPTED
st.session_state is Streamlit's container for session values. It stores values across reruns within a user's Streamlit session. It is useful for preserving a quiz score, a current question number, login information, and chat history, so users retain their progress when the script reruns. Initialize a key only if it does not already exist so a rerun does not reset the saved value.

2. [streamlit_basics.txt:paragraph-4] | distance=0.8244 | ACCEPTED
st.cache_data caches the results of data-returning functions, such as loading a dataset or fetching API data. Reusing a cached result can reduce repeated work. A function's clear method or st.cache_data.clear can clear the corresponding cache, but clearing cached data does not automatically reset widget values in st.session_state.

3. [embeddings_and_search.txt:paragraph-4] | distance=0.8320 | ACCEPTED
ChromaDB stores embeddings together with text, unique IDs, and metadata. PersistentClient saves the database to a specified folder so its data survives process restarts. A collection query returns nearest chunks, and source metadata helps show where each retrieved chunk came from.

Accepted chunks: 3. Checking question and generating response...

Structured response:
{
  "answer": "What do you want it to remember things for?",
  "sources": [
    "[streamlit_basics.txt:paragraph-2]",
    "[streamlit_basics.txt:paragraph-4]",
    "[embeddings_and_search.txt:paragraph-4]"
  ],
  "confidence": "medium",
  "chunks_retrieved": 3
}
```

## Run summary

Queries executed: 4/4; execution errors: 0.

Review all four responses against their expected behavior before submission. Confidence describes the best accepted distance, not a probability that the answer is correct. Similarity alone does not prove answerability.
