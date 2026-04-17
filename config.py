"""Shared settings for the KnowFlow Streamlit app.

The values here are intentionally small and conservative so the app remains
easy to run on a laptop or Streamlit Community Cloud during a hackathon demo.
"""

from pathlib import Path


APP_NAME = "KnowFlow"
APP_TAGLINE = "A source-grounded AI knowledge workspace for study and research."

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / ".knowflow_chroma"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
SAMPLE_FILE = SAMPLE_DATA_DIR / "knowflow_demo.txt"

COLLECTION_NAME = "knowflow_sources"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 950
CHUNK_OVERLAP = 160
DEFAULT_TOP_K = 5

MAX_CONTEXT_CHARS = 9000
MAX_FULL_TEXT_CHARS = 45000

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_TIMEOUT_SECONDS = 20

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
