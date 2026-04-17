"""RAG ingestion, retrieval, and answer generation for KnowFlow."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    MAX_CONTEXT_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)


@dataclass
class SourceChunk:
    """One retrieved piece of evidence from a source document."""

    text: str
    filename: str
    chunk_id: int
    score: float | None = None


class SimpleHashEmbeddingModel:
    """Tiny deterministic embedding fallback when sentence-transformers fails."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: list[str] | str, normalize_embeddings: bool = True) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return [self._embed_one(text, normalize_embeddings) for text in texts]

    def _embed_one(self, text: str, normalize: bool) -> list[float]:
        vector = [0.0] * self.dimensions
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
        for word in words:
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        if normalize:
            norm = sum(value * value for value in vector) ** 0.5
            if norm:
                vector = [value / norm for value in vector]
        return vector


def get_embedding_model() -> tuple[Any, str]:
    """Load the preferred embedding model, with a safe local fallback."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL_NAME), EMBEDDING_MODEL_NAME
    except Exception:
        return SimpleHashEmbeddingModel(), "deterministic keyword fallback"


def get_chroma_collection():
    """Open the persistent ChromaDB collection used by the app."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def make_source_id(filename: str, text: str) -> str:
    """Create a stable id so repeated uploads do not duplicate chunks."""
    digest = hashlib.sha256((filename + "\n" + text).encode("utf-8")).hexdigest()
    return digest[:16]


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into readable overlapping chunks."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", clean)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue

        if current:
            chunks.append(current)
        if len(sentence) > chunk_size:
            step = max(1, chunk_size - overlap)
            for start in range(0, len(sentence), step):
                chunks.append(sentence[start : start + chunk_size])
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk.strip()) > 40]


def index_document(filename: str, text: str, embedding_model: Any | None = None) -> dict[str, Any]:
    """Embed and store a document in ChromaDB."""
    if not text.strip():
        return {"ok": False, "message": "No text was available to index."}

    try:
        model = embedding_model or get_embedding_model()[0]
        collection = get_chroma_collection()
        source_id = make_source_id(filename, text)
        chunks = split_text_into_chunks(text)
        if not chunks:
            return {"ok": False, "message": "The document did not contain enough text to index."}

        existing = collection.get(where={"source_id": source_id}, limit=1)
        if existing.get("ids"):
            return {
                "ok": True,
                "message": f"{filename} was already indexed.",
                "source_id": source_id,
                "chunks": len(chunks),
                "duplicate": True,
            }

        ids = [f"{source_id}-{index}" for index in range(len(chunks))]
        embeddings = model.encode(chunks, normalize_embeddings=True)
        metadatas = [
            {"filename": filename, "chunk_id": index, "source_id": source_id}
            for index in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        return {
            "ok": True,
            "message": f"Indexed {filename}.",
            "source_id": source_id,
            "chunks": len(chunks),
            "duplicate": False,
        }
    except Exception:
        return {"ok": False, "message": "Indexing failed. Try a smaller or cleaner document."}


def retrieve_relevant_chunks(query: str, embedding_model: Any, top_k: int = DEFAULT_TOP_K) -> list[SourceChunk]:
    """Retrieve the most relevant chunks for a user question."""
    if not query.strip():
        return []
    try:
        collection = get_chroma_collection()
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: list[SourceChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                SourceChunk(
                    text=document,
                    filename=metadata.get("filename", "Unknown source"),
                    chunk_id=int(metadata.get("chunk_id", 0)),
                    score=float(distance) if distance is not None else None,
                )
            )
        return chunks
    except Exception:
        return []


def get_indexed_sources() -> list[dict[str, Any]]:
    """Return source names and chunk counts currently stored in ChromaDB."""
    try:
        collection = get_chroma_collection()
        data = collection.get(include=["metadatas"])
        sources: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas", []):
            filename = metadata.get("filename", "Unknown source")
            source_id = metadata.get("source_id", filename)
            if source_id not in sources:
                sources[source_id] = {"filename": filename, "chunks": 0}
            sources[source_id]["chunks"] += 1
        return sorted(sources.values(), key=lambda item: item["filename"].lower())
    except Exception:
        return []


def get_all_document_text(max_chars: int = 45000) -> str:
    """Collect indexed chunks into one text block for tools like summaries."""
    try:
        collection = get_chroma_collection()
        data = collection.get(include=["documents", "metadatas"])
        paired = list(zip(data.get("documents", []), data.get("metadatas", [])))
        paired.sort(key=lambda item: (item[1].get("filename", ""), item[1].get("chunk_id", 0)))
        text = "\n\n".join(document for document, _metadata in paired)
        return text[:max_chars]
    except Exception:
        return ""


def build_context(chunks: list[SourceChunk]) -> str:
    """Format retrieved chunks for a model prompt or fallback answer."""
    parts = []
    total = 0
    for index, chunk in enumerate(chunks, start=1):
        snippet = chunk.text.strip()
        total += len(snippet)
        if total > MAX_CONTEXT_CHARS:
            break
        parts.append(f"[Source {index}: {chunk.filename}, chunk {chunk.chunk_id}]\n{snippet}")
    return "\n\n".join(parts)


def ollama_is_available() -> bool:
    """Check whether a local Ollama server is reachable."""
    try:
        import requests

        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.ok
    except Exception:
        return False


def generate_with_ollama(question: str, chunks: list[SourceChunk]) -> str | None:
    """Ask Ollama for a grounded answer. Returns None if anything fails."""
    if not chunks or not ollama_is_available():
        return None
    context = build_context(chunks)
    prompt = f"""
You are KnowFlow, a careful study assistant. Answer ONLY from the context below.
If the context does not contain enough evidence, say that the uploaded sources do not answer it.
Cite sources inline like [Source 1].

Context:
{context}

Question: {question}
Answer:
""".strip()
    try:
        import requests

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if not response.ok:
            return None
        answer = response.json().get("response", "").strip()
        return answer or None
    except Exception:
        return None


def extractive_answer(question: str, chunks: list[SourceChunk]) -> str:
    """Create a conservative answer by selecting sentences from retrieved chunks."""
    if not chunks:
        return "I could not find relevant evidence in the uploaded sources."

    query_terms = {
        word.lower()
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", question)
        if len(word) > 3
    }
    scored_sentences: list[tuple[int, str, int]] = []
    for source_index, chunk in enumerate(chunks, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 35:
                continue
            words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", sentence.lower()))
            score = len(query_terms & words)
            if score:
                scored_sentences.append((score, sentence, source_index))

    if not scored_sentences:
        first = chunks[0].text.strip()
        preview = first[:650] + ("..." if len(first) > 650 else "")
        return (
            "The retrieved sources are related, but I could not find a sentence that directly "
            f"answers the question. Most relevant excerpt: {preview} [Source 1]"
        )

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = scored_sentences[:4]
    bullets = [f"- {sentence} [Source {source_index}]" for _score, sentence, source_index in selected]
    return "Here is the best source-grounded answer I can extract:\n\n" + "\n".join(bullets)


def answer_question(question: str, chunks: list[SourceChunk]) -> tuple[str, str]:
    """Answer with Ollama when available, otherwise use the deterministic fallback."""
    ollama_answer = generate_with_ollama(question, chunks)
    if ollama_answer:
        return ollama_answer, "Ollama grounded generation"
    return extractive_answer(question, chunks), "Extractive fallback"
