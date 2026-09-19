from sentence_transformers import SentenceTransformer
import numpy as np


# A 500+ word document covering several concepts from the course.
DOCUMENT = """
FastAPI is a modern Python framework for building web APIs. An API allows two programs to communicate through requests and responses. In a FastAPI project, decorators such as @app.get and @app.post connect Python functions to HTTP routes. Path parameters identify a specific resource, while query parameters provide optional filtering or sorting. Pydantic schemas describe the expected shape and data types of request bodies. FastAPI uses those schemas to validate incoming data automatically and returns a helpful 422 response when the data is invalid. The framework also creates interactive Swagger documentation at the /docs route, which makes every endpoint easier to test during development.

A well-organized API separates responsibilities into different files. Database models describe how information is stored, while Pydantic schemas describe data entering and leaving the API. Routers group related endpoints so that a growing application does not place everything in main.py. A service or CRUD layer can hold reusable database operations. SQLAlchemy maps Python classes to database tables and makes it possible to create, read, update, and delete records without writing every SQL statement by hand. Relationships connect tables such as users, students, courses, or tasks. During testing, an in-memory SQLite database can isolate test data and keep the real development database unchanged.

Security is essential when an API stores user information. Passwords should be hashed with a library such as Passlib instead of being saved as readable text. After a successful login, the server can issue a JSON Web Token, commonly called a JWT. The frontend includes that token in the Authorization header when requesting a protected endpoint. The backend verifies the signature and expiration before allowing access. CORS middleware controls which browser origins may call the API, and rate limiting helps slow repeated login attempts or excessive requests. Input constraints such as maximum string lengths and reasonable numeric ranges provide another layer of protection against invalid or abusive data.

Streamlit makes it possible to build a Python frontend without writing a large amount of JavaScript. A Streamlit dashboard can send HTTP requests to a FastAPI backend, display metrics, show records in a table, and collect new data through forms. Session state preserves values such as a login token, chat messages, or a quiz score across reruns. Forms are useful because several inputs can be submitted together instead of triggering the application after every change. Cached functions can prevent an API or dataset from being loaded unnecessarily. A clear interface should also show loading indicators, useful error messages, and confirmation after an action succeeds.

Git records changes to a project and GitHub stores a remote copy that can be shared or submitted. A normal workflow begins with git init for a new local repository, followed by git add, git commit, and git push. Before pushing to a new GitHub repository, the remote address is added with git remote add origin. The command git status shows tracked, modified, and untracked files, which is often the best first step when a file appears to be missing. A .gitignore file prevents virtual environments, secret files, and generated cache folders from being committed. Small commits with descriptive messages make mistakes easier to understand and reverse.

Semantic search compares meaning instead of requiring exact keyword matches. An embedding model converts a sentence, paragraph, or query into a numeric vector. Documents with similar meanings produce vectors that point in similar directions, and cosine similarity measures that relationship. A search system embeds its document collection in advance, embeds the user's query, calculates similarity scores, and returns the highest-ranking results. Similarity thresholds control how many results are accepted: a low threshold improves recall but may include unrelated material, while a high threshold improves precision but can miss useful matches. Testing several realistic queries is necessary because one threshold rarely behaves perfectly for every topic.

Chunking determines the pieces of a document that are embedded and retrieved. Fixed-size chunking creates pieces with a consistent character count, which is simple and predictable. Overlap repeats part of one chunk in the next so that an idea near a boundary is less likely to disappear. However, a fixed boundary can still cut through a sentence or separate an explanation from its supporting details. Paragraph-based chunking preserves the author's natural topic boundaries and usually returns passages that are easier for a person to read. Its weakness is that paragraph lengths can vary widely. Effective retrieval systems select a strategy based on document structure, model limits, and the kind of answers users need.
""".strip()


QUERIES = [
    "How does FastAPI validate request data?",
    "How is a JWT used to protect API endpoints?",
    "Why is overlap useful when chunking a document?",
]


def fixed_size_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks


def paragraph_chunks(text: str) -> list[str]:
    """Split text at blank lines so each paragraph remains intact."""
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> np.ndarray:
    """Create normalized embeddings for a list of chunks."""
    return model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def find_top_results(
    query: str,
    model: SentenceTransformer,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    top_k: int = 2,
) -> list[tuple[float, str]]:
    """Return the chunks with the highest cosine similarity scores."""
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Normalized vectors make the dot product equal cosine similarity.
    scores = chunk_embeddings @ query_embedding
    best_indices = np.argsort(scores)[::-1][:top_k]

    return [(float(scores[index]), chunks[index]) for index in best_indices]


def display_results(strategy: str, results: list[tuple[float, str]]) -> None:
    """Print two ranked results for one chunking strategy."""
    print(f"\n  {strategy}")
    for rank, (score, chunk) in enumerate(results, start=1):
        clean_chunk = " ".join(chunk.split())
        print(f"    {rank}. Score: {score:.4f}")
        print(f"       {clean_chunk}")


def main() -> None:
    """Create both chunk sets, embed them, and compare their searches."""
    fixed_chunks = fixed_size_chunks(DOCUMENT, chunk_size=300, overlap=50)
    paragraphs = paragraph_chunks(DOCUMENT)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    fixed_embeddings = embed_chunks(model, fixed_chunks)
    paragraph_embeddings = embed_chunks(model, paragraphs)

    print("\nCHUNK COUNTS")
    print(f"  Fixed-size (300 characters, 50 overlap): {len(fixed_chunks)}")
    print(f"  Paragraph-based: {len(paragraphs)}")

    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f'Query: "{query}"')

        fixed_results = find_top_results(
            query, model, fixed_chunks, fixed_embeddings
        )
        paragraph_results = find_top_results(
            query, model, paragraphs, paragraph_embeddings
        )

        display_results("FIXED-SIZE TOP 2", fixed_results)
        display_results("PARAGRAPH-BASED TOP 2", paragraph_results)

    print("\n" + "=" * 80)
    print("WRITTEN COMPARISON")
    print(
        "Paragraph-based chunking performed better overall for this document. "
        "Its results preserve complete ideas, so the retrieved passages provide "
        "enough context to answer each query and are easier to read. Fixed-size "
        "chunking sometimes finds a more narrowly focused match, and its "
        "50-character overlap helps protect ideas at chunk boundaries, but several "
        "results begin or end in the middle of a sentence. Paragraph chunking is "
        "the stronger choice here because the document has clear, topic-focused "
        "paragraphs. Fixed-size chunks would be more useful for text without "
        "reliable paragraph breaks or when consistent chunk lengths are required."
    )


if __name__ == "__main__":
    main()
