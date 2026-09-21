# Course document search

This project extends the Guided Example. It contains five course-topic files in `docs/`, a persistent ChromaDB collection, a sidebar source filter, result counts, previews with expandable full text, colored relevance labels, and a unique source-file count.

## Run in PowerShell

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run search_app.py
```

In the browser, click **Re-index Documents** in the sidebar. The first index may take longer because Chroma's default embedding model needs to download once. After indexing, search for something like `How are invalid API requests handled?` or `How does Streamlit remember a quiz score?`.

Select one or more files under **Filter by source file** to restrict search to those sources. Leave it empty to search every file. The result heading shows how many results are visible out of all indexed chunks. With a filter active, the caption also shows how many chunks belong to those selected files. Open **Read full chunk** under a result to see its complete text.

Each nonempty paragraph in a `.txt` or `.md` file becomes one chunk. The five included files create **23 chunks**. To update or add files, edit `docs/` and click **Re-index Documents** again; removed files' old chunks are deleted. Chroma stores its persistent data in `search_db/` (created automatically).

The green, yellow, and red relevance indicators use illustrative Chroma distance thresholds, not calibrated probabilities. Smaller distances mean closer vector matches.
