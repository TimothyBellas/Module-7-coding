from pathlib import Path
from typing import Any

import chromadb


COLLECTION_NAME = "my_knowledge"
DATABASE_PATH = Path(__file__).resolve().parent / "chroma_data"


# Twenty documents from four completed course modules. Each item has a unique
# ID plus the required module and topic metadata.
COURSE_DOCUMENTS = [
    {
        "id": "module3_primary_keys",
        "text": (
            "A primary key uniquely identifies every row in a database table, "
            "while a foreign key connects a row to a related table."
        ),
        "metadata": {"module": "3", "topic": "database"},
    },
    {
        "id": "module3_sql_joins",
        "text": (
            "SQL joins combine related rows from multiple tables. INNER JOIN "
            "returns matches, while LEFT JOIN also keeps unmatched left-side rows."
        ),
        "metadata": {"module": "3", "topic": "database"},
    },
    {
        "id": "module3_sqlalchemy_models",
        "text": (
            "SQLAlchemy models are Python classes that map attributes to database "
            "table columns and allow records to be managed as Python objects."
        ),
        "metadata": {"module": "3", "topic": "database"},
    },
    {
        "id": "module3_relationships",
        "text": (
            "SQLAlchemy relationships connect models such as students and courses "
            "and make related records accessible through object attributes."
        ),
        "metadata": {"module": "3", "topic": "database"},
    },
    {
        "id": "module3_crud",
        "text": (
            "CRUD represents the four fundamental database operations: create, "
            "read, update, and delete stored records."
        ),
        "metadata": {"module": "3", "topic": "database"},
    },
    {
        "id": "module4_http_requests",
        "text": (
            "The Python requests library sends HTTP requests to external APIs, and "
            "response.json() converts a JSON response into Python data."
        ),
        "metadata": {"module": "4", "topic": "api"},
    },
    {
        "id": "module4_request_parts",
        "text": (
            "An HTTP request can contain a method, URL, headers, query parameters, "
            "path parameters, and a body containing submitted data."
        ),
        "metadata": {"module": "4", "topic": "api"},
    },
    {
        "id": "module4_status_codes",
        "text": (
            "HTTP status codes describe request outcomes: 200 means success, 201 "
            "means created, 404 means not found, and 500 means a server error."
        ),
        "metadata": {"module": "4", "topic": "api"},
    },
    {
        "id": "module4_api_authentication",
        "text": (
            "API credentials are commonly sent in request headers so a server can "
            "authenticate the caller without placing secrets in the URL."
        ),
        "metadata": {"module": "4", "topic": "security"},
    },
    {
        "id": "module4_rest_design",
        "text": (
            "REST APIs use resource-based URLs and HTTP methods such as GET, POST, "
            "PUT, PATCH, and DELETE to express operations clearly."
        ),
        "metadata": {"module": "4", "topic": "api"},
    },
    {
        "id": "module5_fastapi_validation",
        "text": (
            "FastAPI uses Pydantic schemas and Python type hints to validate request "
            "data and automatically document endpoints in Swagger UI."
        ),
        "metadata": {"module": "5", "topic": "api"},
    },
    {
        "id": "module5_routers",
        "text": (
            "FastAPI APIRouter objects organize related endpoints into separate "
            "files, keeping larger applications modular and maintainable."
        ),
        "metadata": {"module": "5", "topic": "api"},
    },
    {
        "id": "module5_dependency_injection",
        "text": (
            "FastAPI dependencies provide reusable logic for database sessions, "
            "authentication checks, and other requirements shared by endpoints."
        ),
        "metadata": {"module": "5", "topic": "api"},
    },
    {
        "id": "module5_jwt",
        "text": (
            "A JSON Web Token carries signed login claims. Protected endpoints "
            "verify the token's signature and expiration before granting access."
        ),
        "metadata": {"module": "5", "topic": "security"},
    },
    {
        "id": "module5_api_protection",
        "text": (
            "CORS restricts allowed browser origins, rate limiting slows excessive "
            "requests, and input constraints reject unreasonable values."
        ),
        "metadata": {"module": "5", "topic": "security"},
    },
    {
        "id": "module6_streamlit_reruns",
        "text": (
            "Streamlit executes the Python script from top to bottom whenever a user "
            "interacts with a widget."
        ),
        "metadata": {"module": "6", "topic": "frontend"},
    },
    {
        "id": "module6_session_state",
        "text": (
            "Streamlit session state preserves information such as authentication "
            "tokens, chat history, and quiz scores across application reruns."
        ),
        "metadata": {"module": "6", "topic": "frontend"},
    },
    {
        "id": "module6_forms",
        "text": (
            "A Streamlit form groups several input widgets and sends their values "
            "together only when the user presses the submit button."
        ),
        "metadata": {"module": "6", "topic": "frontend"},
    },
    {
        "id": "module6_caching",
        "text": (
            "The st.cache_data decorator avoids repeating expensive data loading or "
            "API calls when the inputs have not changed."
        ),
        "metadata": {"module": "6", "topic": "frontend"},
    },
    {
        "id": "module6_full_stack",
        "text": (
            "A Streamlit frontend can call a FastAPI backend with HTTP requests, "
            "display returned records, and submit new data through forms."
        ),
        "metadata": {"module": "6", "topic": "frontend"},
    },
]


# PersistentClient writes the database to disk instead of keeping it in memory.
client = chromadb.PersistentClient(path=str(DATABASE_PATH))

# This returns the existing collection on later runs instead of creating a
# conflicting duplicate.
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def populate_knowledge_base() -> None:
    """Insert or update every course document without duplicate-ID errors."""
    collection.upsert(
        ids=[item["id"] for item in COURSE_DOCUMENTS],
        documents=[item["text"] for item in COURSE_DOCUMENTS],
        metadatas=[item["metadata"] for item in COURSE_DOCUMENTS],
    )


def search(query: str, module_filter: str | None = None) -> list[dict[str, Any]]:
    """Return the five closest documents, optionally limited to one module."""
    query_options: dict[str, Any] = {
        "query_texts": [query],
        "n_results": 5,
        "include": ["documents", "metadatas", "distances"],
    }

    if module_filter is not None:
        query_options["where"] = {"module": str(module_filter)}

    raw_results = collection.query(**query_options)

    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    return [
        {
            "id": document_id,
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]


def display_results(
    label: str,
    query: str,
    module_filter: str | None = None,
) -> None:
    """Run one search and display its five ranked results."""
    print("\n" + "=" * 80)
    print(label)
    print(f'Query: "{query}"')
    print(f"Module filter: {module_filter or 'None'}")

    results = search(query, module_filter)

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n{rank}. Distance: {result['distance']:.4f} | "
            f"ID: {result['id']}"
        )
        print(f"   Metadata: module={metadata['module']}, topic={metadata['topic']}")
        print(f"   Text: {result['text']}")


def main() -> None:
    """Populate the persistent collection and demonstrate three searches."""
    print(f"Persistent database folder: {DATABASE_PATH}")
    print(f'Preparing collection: "{COLLECTION_NAME}"')

    populate_knowledge_base()
    print(f"Collection contains {collection.count()} documents.")

    # 1. Broad query without a metadata filter.
    display_results(
        label="TEST 1: BROAD, UNFILTERED SEARCH",
        query="How can I build and protect a web API?",
    )

    # 2. Search only documents whose metadata says they came from Module 3.
    display_results(
        label="TEST 2: SEARCH FILTERED TO MODULE 3",
        query="How can I connect information stored in separate tables?",
        module_filter="3",
    )

    # 3. A semantic query deliberately phrased differently from the saved text.
    display_results(
        label="TEST 3: DIFFERENT WORDING, NO FILTER",
        query="How can information stick around when the page redraws?",
    )


if __name__ == "__main__":
    main()
