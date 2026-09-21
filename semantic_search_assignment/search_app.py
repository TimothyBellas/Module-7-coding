"""Search course notes stored in a persistent ChromaDB collection."""

from pathlib import Path

import chromadb
import streamlit as st


st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"
DB_DIR = PROJECT_DIR / "search_db"


@st.cache_resource
def get_collection():
    """Keep one persistent Chroma client and collection across Streamlit reruns."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name="course_docs")


def load_and_chunk(directory: Path):
    """Make one searchable chunk per nonempty paragraph in each text file."""
    chunks = []
    if not directory.is_dir():
        return chunks

    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue

        content = path.read_text(encoding="utf-8")
        paragraphs = [paragraph.strip() for paragraph in content.split("\n\n")]
        for index, paragraph in enumerate(p for p in paragraphs if p):
            chunks.append(
                {
                    "id": f"{path.name}_{index}",
                    "text": paragraph,
                    "source": path.name,
                    "chunk_index": index,
                }
            )
    return chunks


def reindex_documents(collection):
    """Upsert the current files and remove chunks from old or deleted files."""
    chunks = load_and_chunk(DOCS_DIR)
    if not chunks:
        return 0, 0

    existing_ids = set(collection.get(include=[])['ids'])
    current_ids = {chunk["id"] for chunk in chunks}

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
            for chunk in chunks
        ],
    )

    stale_ids = sorted(existing_ids - current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)

    return len(chunks), len({chunk["source"] for chunk in chunks})


collection = get_collection()

with st.sidebar:
    st.title("📁 Document Manager")

    if st.button("🔄 Re-index Documents", use_container_width=True):
        with st.spinner("Indexing documents..."):
            chunk_count, file_count = reindex_documents(collection)
        if chunk_count:
            st.success(f"Indexed {chunk_count} chunks from {file_count} files.")
        else:
            st.warning("Add .txt or .md files to docs/ before indexing.")

    total_documents = collection.count()
    # Read the indexed metadata so the filter includes only files actually in the DB.
    indexed_metadata = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted(
        {item["source"] for item in indexed_metadata if item and "source" in item}
    )

    st.metric("Documents in DB", total_documents)
    st.metric("Unique source files", len(sources))  # Bonus feature
    st.divider()

    selected_sources = st.multiselect(
        "Filter by source file",
        options=sources,
        help="Leave empty to search all indexed files.",
    )
    n_results = st.slider("Results to show", min_value=1, max_value=10, value=5)

st.title("🔍 Semantic Search")
st.write("Search course documents by meaning, then open a result to read its full text.")
query = st.text_input("Enter your search query", placeholder="How does authentication work?")

if total_documents == 0:
    st.info("👈 Click 'Re-index Documents' in the sidebar to load the sample files first.")
elif query.strip():
    # Chroma applies this metadata filter before ranking results. No selection means all files.
    source_filter = {"source": {"$in": selected_sources}} if selected_sources else None
    matching_documents = (
        len(collection.get(where=source_filter, include=[])["ids"])
        if source_filter
        else total_documents
    )

    if matching_documents == 0:
        st.write(f"Showing 0 of {total_documents} total documents")
        st.info("No indexed chunks belong to the selected source files.")
    else:
        results = collection.query(
            query_texts=[query.strip()],
            n_results=min(n_results, matching_documents),
            where=source_filter,
            include=["documents", "metadatas", "distances"],
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        st.subheader(f"Showing {len(documents)} of {total_documents} total documents")
        if selected_sources:
            st.caption(f"{matching_documents} chunks match the selected source files.")

        for document, metadata, distance in zip(documents, metadatas, distances):
            # Lower Chroma distances are closer matches; thresholds are illustrative.
            if distance < 0.5:
                relevance = "🟢 High"
            elif distance < 1.0:
                relevance = "🟡 Medium"
            else:
                relevance = "🔴 Low"

            with st.container():
                col_source, col_relevance = st.columns([3, 1])
                with col_source:
                    st.write(
                        f"**{metadata['source']}** — chunk {metadata['chunk_index']}"
                    )
                with col_relevance:
                    st.write(f"{relevance} (dist: {distance:.3f})")

                preview = document[:150].rstrip()
                st.write(preview + ("…" if len(document) > 150 else ""))
                with st.expander("Read full chunk"):
                    st.text(document)
                st.divider()
