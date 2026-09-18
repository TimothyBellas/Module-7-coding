"""Compare semantic-search results at several similarity thresholds."""

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLDS = (0.3, 0.5, 0.7)

# The knowledge base contains 16 sentences across four distinct topics.
KNOWLEDGE_BASE = [
    {
        "id": "python_deploy",
        "topic": "Python",
        "text": "Deploy a FastAPI application with Uvicorn and a Docker container.",
    },
    {
        "id": "python_environment",
        "topic": "Python",
        "text": "A Python virtual environment isolates each project's dependencies.",
    },
    {
        "id": "python_testing",
        "topic": "Python",
        "text": "Pytest runs automated tests and reusable setup fixtures for Python code.",
    },
    {
        "id": "python_database",
        "topic": "Python",
        "text": "SQLAlchemy connects Python applications to relational databases.",
    },
    {
        "id": "cooking_bread",
        "topic": "Cooking",
        "text": "Bread develops a crisp crust when baked in a properly preheated oven.",
    },
    {
        "id": "cooking_sauce",
        "topic": "Cooking",
        "text": "Simmering a sauce slowly concentrates its flavor without burning it.",
    },
    {
        "id": "cooking_knife",
        "topic": "Cooking",
        "text": "A sharp chef's knife makes vegetable preparation safer and more precise.",
    },
    {
        "id": "cooking_timing",
        "topic": "Cooking",
        "text": "Careful timing and temperature control are essential for consistent cooking results.",
    },
    {
        "id": "space_orbit",
        "topic": "Space",
        "text": "Astronauts appear weightless because they are continuously falling around Earth in orbit.",
    },
    {
        "id": "space_mars",
        "topic": "Space",
        "text": "Mars rovers study rocks and soil for evidence of ancient water.",
    },
    {
        "id": "space_telescope",
        "topic": "Space",
        "text": "Space telescopes observe distant galaxies without interference from Earth's atmosphere.",
    },
    {
        "id": "space_radio",
        "topic": "Space",
        "text": "Radio signals carry commands and scientific data between Earth and spacecraft.",
    },
    {
        "id": "music_guitar",
        "topic": "Music",
        "text": "Daily guitar practice builds finger strength and improves chord changes.",
    },
    {
        "id": "music_rhythm",
        "topic": "Music",
        "text": "Practicing with a metronome develops accurate rhythm and musical timing.",
    },
    {
        "id": "music_chords",
        "topic": "Music",
        "text": "Chord progressions create harmony and shape the emotional direction of a song.",
    },
    {
        "id": "music_recording",
        "topic": "Music",
        "text": "A microphone converts a musical performance into an audio signal for recording.",
    },
]

# Relevant IDs are human-selected answers used to reveal false negatives.
# The final query is intentionally ambiguous between cooking and music.
TEST_QUERIES = [
    {
        "text": "How can I put my Python web service online?",
        "relevant_ids": {"python_deploy", "python_environment"},
    },
    {
        "text": "What helps me bake a loaf with a good crust?",
        "relevant_ids": {"cooking_bread", "cooking_timing"},
    },
    {
        "text": "Why do people float inside an orbiting spacecraft?",
        "relevant_ids": {"space_orbit"},
    },
    {
        "text": "How can I get better at changing chords on guitar?",
        "relevant_ids": {"music_guitar", "music_chords", "music_rhythm"},
    },
    {
        "text": "How can careful timing improve the final performance?",
        "relevant_ids": {"cooking_timing", "music_rhythm"},
    },
]


def score_documents(model, document_embeddings, query_text):
    """Return every knowledge-base document sorted by cosine similarity."""
    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    similarity_scores = util.cos_sim(query_embedding, document_embeddings)[0]

    results = []
    for document, score in zip(KNOWLEDGE_BASE, similarity_scores):
        results.append(
            {
                "id": document["id"],
                "topic": document["topic"],
                "text": document["text"],
                "score": score.item(),
            }
        )

    return sorted(results, key=lambda result: result["score"], reverse=True)


def display_threshold_results(results, relevant_ids, threshold):
    """Display passing results and relevant false negatives for one threshold."""
    passing_results = [
        result for result in results if result["score"] >= threshold
    ]
    missed_relevant_results = [
        result
        for result in results
        if result["id"] in relevant_ids and result["score"] < threshold
    ]

    noun = "result" if len(passing_results) == 1 else "results"
    print(f"\n  Threshold {threshold:.1f}: {len(passing_results)} {noun}")

    if passing_results:
        for result in passing_results:
            relevance_marker = (
                " <-- expected relevant" if result["id"] in relevant_ids else ""
            )
            print(
                f"    [{result['score']:.4f}] "
                f"({result['topic']}) {result['text']}"
                f"{relevance_marker}"
            )
    else:
        print("    No documents passed this threshold.")

    if missed_relevant_results:
        print("    Relevant results missed at this threshold:")
        for result in missed_relevant_results:
            print(
                f"      [{result['score']:.4f}] "
                f"({result['topic']}) {result['text']}"
            )
    else:
        print("    Relevant results missed at this threshold: None")


def main():
    """Run all five queries at the three required thresholds."""
    print("Loading the sentence-transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    document_embeddings = model.encode(
        [document["text"] for document in KNOWLEDGE_BASE],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("\nSEMANTIC SEARCH THRESHOLD EXPERIMENT")
    print("=" * 80)

    for query in TEST_QUERIES:
        results = score_documents(model, document_embeddings, query["text"])
        print(f'\nQuery: "{query["text"]}"')

        for threshold in THRESHOLDS:
            display_threshold_results(
                results,
                query["relevant_ids"],
                threshold,
            )

        print("\n" + "-" * 80)

    print(
        "\nConclusion: Lower thresholds return more results but may include noise; "
        "higher thresholds are more selective but can miss useful matches."
    )


if __name__ == "__main__":
    main()
