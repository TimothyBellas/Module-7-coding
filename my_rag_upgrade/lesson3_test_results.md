# RAG test results

Run time (UTC): 2026-09-30T21:14:22+00:00

Ollama model: llama3.2:3b

Stored paragraph chunks: 24

These are actual outputs from this run. Expected behavior is a review guide, not a generated answer or an automatic quality score.

## Test 1: Answerable from the documents

**Question:** What is st.session_state, and why is it useful in Streamlit?

**Expected behavior:** Explain that st.session_state preserves values between reruns within a user's session; cite streamlit_basics.txt.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved chunks (lower cosine distance = closer match):

1. [streamlit_basics.txt:paragraph-2] | distance=0.1988
st.session_state is Streamlit's container for session values. It stores values across reruns within a user's Streamlit session. It is useful for preserving a quiz score, a current question number, login information, and chat history, so users retain their progress when the script reruns. Initialize a key only if it does not already exist so a rerun does not reset the saved value.

2. [streamlit_basics.txt:paragraph-4] | distance=0.5175
st.cache_data caches the results of data-returning functions, such as loading a dataset or fetching API data. Reusing a cached result can reduce repeated work. A function's clear method or st.cache_data.clear can clear the corresponding cache, but clearing cached data does not automatically reset widget values in st.session_state.

3. [streamlit_basics.txt:paragraph-1] | distance=0.6240
Streamlit runs a Python script from top to bottom when a user interacts with a widget. This rerun model makes it straightforward to rebuild the page, but ordinary local variables do not preserve their values between separate script executions.

Generating answer with Ollama...

Answer:
st.session_state is Streamlit's container for session values. It stores values across reruns within a user's Streamlit session. It is useful for preserving a quiz score, a current question number, login information, and chat history, so users retain their progress when the script reruns. [streamlit_basics.txt:paragraph-2]
```

## Test 2: Related, but not directly documented

**Question:** How do I deploy a FastAPI application to AWS Lambda?

**Expected behavior:** Acknowledge that these documents do not explain AWS Lambda deployment. Any supported FastAPI background must have a citation; do not invent deployment steps.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved chunks (lower cosine distance = closer match):

1. [fastapi_basics.txt:paragraph-3] | distance=0.4925
FastAPI generates interactive API documentation at /docs and alternative documentation at /redoc. Run a development application whose file is main.py and whose FastAPI instance is named app with python -m uvicorn main:app --reload.

2. [fastapi_basics.txt:paragraph-1] | distance=0.5358
FastAPI is a Python framework for building web APIs. An endpoint combines a URL path with an HTTP method. For example, GET /tasks can retrieve tasks, while POST /tasks can create a task. CRUD means create, read, update, and delete.

3. [fastapi_basics.txt:paragraph-2] | distance=0.6163
FastAPI uses Pydantic models to validate request data. A request schema describes expected fields and types. Field constraints can set a minimum number or a maximum string length. Invalid request data normally produces an HTTP 422 validation response.

Generating answer with Ollama...

Answer:
I don't have enough information in the provided documents to answer that. The topic of deploying a FastAPI application to AWS Lambda is not mentioned in the context.
```

## Test 3: Completely outside the document scope

**Question:** What is the best recipe for chocolate brownies?

**Expected behavior:** Say that the provided documents do not contain enough information; do not supply a recipe or cite unrelated course notes.

**Execution status:** LLM answer produced. Review its accuracy and citations below.

**Observed output:**

```text

Retrieved chunks (lower cosine distance = closer match):

1. [sqlalchemy_basics.txt:paragraph-1] | distance=0.9201
SQLAlchemy is a Python library for working with relational databases. Its object-relational mapper maps Python model classes to database tables. An instance of a mapped class represents a row, and mapped attributes represent columns.

2. [sqlalchemy_basics.txt:paragraph-3] | distance=0.9414
A SQLAlchemy session manages work with database records. Add a new model instance to the session and call commit to persist the transaction. Refresh can reload database-generated values. Rollback cancels pending changes after a failed transaction.

3. [embeddings_and_search.txt:paragraph-4] | distance=0.9523
ChromaDB stores embeddings together with text, unique IDs, and metadata. PersistentClient saves the database to a specified folder so its data survives process restarts. A collection query returns nearest chunks, and source metadata helps show where each retrieved chunk came from.

Generating answer with Ollama...

Answer:
I don't have enough information in the provided documents to answer that. The requested topic is "best recipe for chocolate brownies," which is not mentioned in the context.
```

## Run summary

LLM answers produced: 3/3.

Review the three answers against their expected behavior before submission. Retrieving a nearest neighbor alone does not prove a question is answerable.
