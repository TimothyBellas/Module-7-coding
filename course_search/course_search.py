"""Interactive semantic search over topics learned in the course."""

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "all-MiniLM-L6-v2"

# These summaries cover concepts practiced throughout the course.
COURSE_SUMMARIES = [
    "FastAPI automatically validates incoming request data using Pydantic models.",
    "Pydantic schemas define the fields, types, and rules that valid data must follow.",
    "Path parameters identify a specific resource while query parameters filter results.",
    "CRUD endpoints let an API create, read, update, and delete stored records.",
    "SQLAlchemy models map Python classes to tables in a relational database.",
    "Database relationships connect records such as students, courses, and enrollments.",
    "JWT authentication protects API routes by verifying a signed access token.",
    "Password hashing keeps user credentials safer than storing plain-text passwords.",
    "Custom exception handlers return consistent and helpful API error responses.",
    "CORS middleware controls which frontend origins may send requests to an API.",
    "Rate limiting helps protect endpoints from too many requests in a short time.",
    "Pytest fixtures create reusable setup data for automated API tests.",
    "An in-memory SQLite database keeps tests isolated from production data.",
    "JavaScript fetch sends HTTP requests and displays API data on a web page.",
    "A Streamlit frontend can communicate with a FastAPI backend through HTTP requests.",
]


def find_top_matches(query, model, sentence_embeddings, top_k=3):
    """Return the most semantically similar course summaries for a query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    result_count = min(top_k, len(COURSE_SUMMARIES))
    top_results = similarity_scores.topk(k=result_count)

    matches = []
    for score, index in zip(top_results.values, top_results.indices):
        matches.append((score.item(), COURSE_SUMMARIES[index.item()]))

    return matches


def main():
    """Load the model once and keep searching until the user types quit."""
    print("Loading semantic search model...")
    model = SentenceTransformer(MODEL_NAME)
    sentence_embeddings = model.encode(
        COURSE_SUMMARIES,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    print("Course search is ready!\n")

    while True:
        try:
            query = input("Search (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a search question or type 'quit'.\n")
            continue

        matches = find_top_matches(query, model, sentence_embeddings)

        print("\nTop 3 results:")
        for number, (score, sentence) in enumerate(matches, start=1):
            print(f"  {number}. [{score:.4f}] {sentence}")
        print()


if __name__ == "__main__":
    main()
