"""Streamlit UI: inspect the index, re-index it, and search source chunks."""

import streamlit as st

from ingest import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ingest
from search import get_collection_stats, search


st.set_page_config(page_title="Course Notes Search", page_icon="🔎", layout="wide")
st.title("🔎 Course Notes Search")
st.caption("Search Python, APIs, databases and AI course notes by meaning.")

stats = get_collection_stats()
with st.sidebar:
    st.header("Index and filters")
    st.metric("Documents indexed", stats["unique_sources"])
    st.metric("Searchable chunks", stats["total_chunks"])
    if stats["chunk_size"]:
        st.caption(f"Current index: {stats['chunk_size']} characters, {stats['overlap']} overlap")

    with st.expander("Re-index settings"):
        chunk_size = st.number_input("Chunk size (characters)", min_value=100,
                                     max_value=2000, value=stats["chunk_size"] or DEFAULT_CHUNK_SIZE,
                                     step=50)
        max_overlap = min(400, int(chunk_size) - 1)
        overlap = st.number_input("Overlap (characters)", min_value=0,
                                  max_value=max_overlap,
                                  value=min(stats["overlap"] if stats["overlap"] is not None
                                            else DEFAULT_OVERLAP, max_overlap), step=10)
        if st.button("Re-index documents", type="primary", use_container_width=True):
            with st.spinner("Reading, embedding, and storing documents..."):
                try:
                    ingest(chunk_size=int(chunk_size), overlap=int(overlap))
                except Exception as error:
                    st.error(f"Indexing failed: {error}")
                else:
                    st.success("Index updated.")
                    st.rerun()

    result_count = st.slider("Maximum results", 1, 15, 5)
    min_similarity = st.slider("Minimum cosine similarity", 0.0, 1.0, 0.0, 0.05,
                               help="0 = no cutoff. Higher values remove weaker matches.")
    selected_sources = st.multiselect("Source files", stats["source_names"],
                                      help="No selection searches every source.")

first, second = st.columns(2)
first.metric("Source files", stats["unique_sources"])
second.metric("Indexed passages", stats["total_chunks"])

if stats["total_chunks"] == 0:
    st.info("No index yet. Run `python ingest.py` or use Re-index documents in the sidebar.")

query = st.text_input("Your question", placeholder="How do SQL joins connect related tables?")
if query:
    if not query.strip():
        st.warning("Enter a question with at least one non-space character.")
    else:
        try:
            with st.spinner("Searching..."):
                results = search(query, n_results=result_count,
                                 sources=selected_sources or None,
                                 distance_threshold=1.0 - min_similarity
                                 if min_similarity > 0 else None)
        except Exception as error:
            st.error(f"Search failed: {error}")
        else:
            if not results:
                st.info("No results match the query and selected filters. Try another question or lower the minimum similarity.")
            else:
                st.subheader(f"Results ({len(results)})")
                st.caption("Score = 1 − cosine distance; scores describe similarity, not answer correctness."
                           " Relevance badges are rough visual guides.")
                for rank, hit in enumerate(results, 1):
                    level = "🟢 Strong" if hit["score"] >= 0.65 else ("🟡 Moderate" if hit["score"] >= 0.4 else "⚪ Weak")
                    with st.container(border=True):
                        st.markdown(f"**{rank}. {hit['source']}** · passage {hit['chunk_index'] + 1}")
                        st.caption(f"{level} · similarity {hit['score']:.3f} · distance {hit['distance']:.3f}")
                        st.write(hit["text"])
