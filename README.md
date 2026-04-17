# KnowFlow

KnowFlow is a NotebookLM-style AI knowledge workspace built with a Python-first stack. It lets users upload PDF or TXT sources, ask source-grounded questions, generate summaries, create quizzes, review flashcards, and extract topics.

The app is designed for hackathon demos: simple to run, free/open-source by default, and stable even when no local LLM is installed.

## Features

- Upload PDF and TXT files.
- Parse documents safely with PyMuPDF and a pdfplumber fallback.
- Chunk documents and store embeddings in a persistent local ChromaDB database.
- Retrieve relevant source chunks for each question.
- Generate grounded answers with optional local Ollama.
- Fall back to deterministic extractive answers when Ollama is unavailable.
- Generate quick, detailed, and exam-style summaries.
- Generate 5-question or 10-question multiple-choice quizzes.
- Generate flashcards and key topic chips.
- Includes a bundled demo document for live presentations.
- Runs locally and can be deployed on Streamlit Community Cloud.

## Folder Structure

```text
knowflow/
├── app.py
├── config.py
├── parser_utils.py
├── rag_pipeline.py
├── quiz_utils.py
├── summary_utils.py
├── requirements.txt
├── README.md
├── sample_data/
│   └── knowflow_demo.txt
└── .streamlit/
    └── config.toml
```

## Run Locally

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Optional Ollama Integration

KnowFlow works without Ollama. If Ollama is not installed or not running, the app uses a deterministic extractive fallback that only quotes and synthesizes from retrieved source chunks.

To use local generative answers:

```bash
ollama pull llama3.1:8b
ollama serve
streamlit run app.py
```

The model name and Ollama URL can be changed in `config.py`.

## Deploy on Streamlit Community Cloud

Streamlit Community Cloud supports free GitHub-based deployment to a public `streamlit.app` URL.

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy.

On Streamlit Community Cloud, Ollama is usually not available. That is okay: KnowFlow automatically uses the extractive fallback path.

## How the RAG Flow Works

1. The user uploads a PDF or TXT file.
2. `parser_utils.py` extracts clean text from the file.
3. `rag_pipeline.py` splits the text into overlapping chunks.
4. The app creates embeddings for each chunk with `sentence-transformers`.
5. Chunks, embeddings, and metadata are stored in local persistent ChromaDB.
6. When the user asks a question, the question is embedded too.
7. ChromaDB returns the most relevant chunks.
8. KnowFlow answers only from those chunks.
9. The app shows the evidence chunks so the user can verify the answer.

## Hackathon Demo Script

1. Start with the one-line pitch: "KnowFlow turns uploaded notes into a source-grounded study assistant."
2. Click "Load demo document" to avoid depending on a fresh upload.
3. Ask: "What does KnowFlow do?"
4. Expand the evidence section to show that answers are tied to sources.
5. Generate an exam summary.
6. Generate a quiz and submit it.
7. Show flashcards and topic chips.
8. Mention that it uses free open-source tools and still works without paid APIs.

## Notes

- Scanned image-only PDFs may not produce text because OCR is intentionally not included in this stable MVP.
- The first run may take time while `sentence-transformers` downloads the embedding model.
- ChromaDB data is stored locally in `.knowflow_chroma/`.
