"""KnowFlow: a NotebookLM-style knowledge workspace built with Streamlit."""

from __future__ import annotations

from html import escape

import streamlit as st

from config import APP_NAME, APP_TAGLINE, DEFAULT_TOP_K, MAX_FULL_TEXT_CHARS, SAMPLE_FILE
from parser_utils import parse_local_text_file, parse_uploaded_file
from quiz_utils import generate_quiz, score_quiz
from rag_pipeline import (
    answer_question,
    get_all_document_text,
    get_embedding_model,
    get_indexed_sources,
    index_document,
    retrieve_relevant_chunks,
)
from summary_utils import extract_topics, generate_flashcards, generate_summary


st.set_page_config(page_title=APP_NAME, page_icon="K", layout="wide")


CUSTOM_CSS = """
<style>
    .stApp {
        background: #101316;
        color: #f5f7fb;
    }
    [data-testid="stSidebar"] {
        background: #151a1f;
        border-right: 1px solid #2b333c;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid #2d3742;
        border-radius: 8px;
        background: linear-gradient(135deg, #172027 0%, #1d262b 55%, #18221d 100%);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .muted {
        color: #aeb8c4;
    }
    .metric-card {
        padding: 0.8rem;
        border: 1px solid #2d3742;
        border-radius: 8px;
        background: #171d22;
    }
    .topic-chip {
        display: inline-block;
        margin: 0.2rem 0.25rem 0.2rem 0;
        padding: 0.35rem 0.6rem;
        border: 1px solid #3c6f5d;
        border-radius: 8px;
        background: #14251f;
        color: #dff8ea;
        font-size: 0.9rem;
    }
</style>
"""


def init_state() -> None:
    """Create Streamlit session keys once."""
    defaults = {
        "messages": [],
        "quiz": [],
        "quiz_submitted": False,
        "workspace_name": "My Knowledge Workspace",
        "embedding_status": "Not loaded",
        "processed_uploads": set(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Cache the embedding model so reruns stay fast."""
    return get_embedding_model()


def index_parsed_document(filename: str, text: str) -> None:
    """Index one parsed document and report a friendly status."""
    with st.spinner(f"Indexing {filename}..."):
        model, model_name = load_embedding_model()
        st.session_state.embedding_status = model_name
        result = index_document(filename, text, model)
    if result.get("ok"):
        st.success(f"{result['message']} Chunks: {result.get('chunks', 0)}")
    else:
        st.error(result.get("message", "Indexing failed."))


def render_sidebar() -> None:
    """Upload controls, source list, and demo helpers."""
    with st.sidebar:
        st.title(APP_NAME)
        st.caption(APP_TAGLINE)
        st.text_input("Workspace name", key="workspace_name")

        uploads = st.file_uploader(
            "Upload sources",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="PDF and TXT files work best. Scanned PDFs may not contain selectable text.",
        )
        if uploads:
            for uploaded_file in uploads:
                upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
                if upload_key in st.session_state.processed_uploads:
                    continue
                parsed = parse_uploaded_file(uploaded_file)
                if parsed.error:
                    st.warning(f"{parsed.filename}: {parsed.error}")
                    continue
                index_parsed_document(parsed.filename, parsed.text)
                st.session_state.processed_uploads.add(upload_key)

        if st.button("Load demo document", use_container_width=True):
            parsed = parse_local_text_file(SAMPLE_FILE)
            if parsed.error:
                st.warning(parsed.error)
            else:
                index_parsed_document(parsed.filename, parsed.text)

        st.divider()
        st.subheader("Indexed Sources")
        sources = get_indexed_sources()
        if not sources:
            st.info("No sources indexed yet. Upload a file or load the demo document.")
        for source in sources:
            st.write(f"- {source['filename']} ({source['chunks']} chunks)")

        total_chunks = sum(source["chunks"] for source in sources)
        st.caption(f"Embedding: {st.session_state.embedding_status}")
        st.caption(f"Total chunks: {total_chunks}")


def render_chat() -> None:
    """Main source-grounded chat interface."""
    st.subheader("Ask Your Sources")
    st.caption(
        "Answers are grounded in retrieved source chunks. If Ollama is not available, "
        "KnowFlow uses a safe extractive fallback."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your uploaded sources")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            model, model_name = load_embedding_model()
            st.session_state.embedding_status = model_name
            chunks = retrieve_relevant_chunks(question, model, DEFAULT_TOP_K)
            if not chunks:
                answer = "I could not find relevant evidence. Add sources or try a more specific question."
                mode = "No retrieval result"
            else:
                answer, mode = answer_question(question, chunks)
            st.markdown(answer)
            st.caption(f"Generation mode: {mode}")

            if chunks:
                with st.expander("Evidence and source chunks"):
                    for index, chunk in enumerate(chunks, start=1):
                        score = f" | distance {chunk.score:.3f}" if chunk.score is not None else ""
                        st.markdown(f"**Source {index}: {chunk.filename}, chunk {chunk.chunk_id}{score}**")
                        st.write(chunk.text)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception:
            st.error("Something went wrong while answering. The app is still running; try a shorter question.")


def render_study_tools() -> None:
    """Summaries, topics, flashcards, and quiz generation."""
    st.subheader("Study Tools")
    full_text = get_all_document_text(MAX_FULL_TEXT_CHARS)
    if not full_text:
        st.info("Upload a source or load the demo document to unlock summaries, quizzes, topics, and flashcards.")
        return

    tab_summary, tab_quiz, tab_flashcards, tab_topics = st.tabs(["Summaries", "Quiz", "Flashcards", "Topics"])

    with tab_summary:
        col1, col2, col3 = st.columns(3)
        if col1.button("Quick Summary", use_container_width=True):
            st.session_state.summary = generate_summary(full_text, "quick")
        if col2.button("Detailed Summary", use_container_width=True):
            st.session_state.summary = generate_summary(full_text, "detailed")
        if col3.button("Exam Summary", use_container_width=True):
            st.session_state.summary = generate_summary(full_text, "exam")
        st.markdown(st.session_state.get("summary", "Choose a summary style."))

    with tab_quiz:
        question_count = st.radio("Number of questions", [5, 10], horizontal=True)
        if st.button("Generate Quiz", use_container_width=True):
            st.session_state.quiz = generate_quiz(full_text, question_count)
            st.session_state.quiz_submitted = False
            if not st.session_state.quiz:
                st.warning("The sources did not contain enough clear concepts to make a quiz.")

        if st.session_state.quiz:
            answers = {}
            with st.form("quiz_form"):
                for index, item in enumerate(st.session_state.quiz):
                    st.markdown(f"**Question {index + 1}**")
                    st.write(item["question"])
                    answers[index] = st.radio(
                        "Choose one",
                        item["options"],
                        key=f"quiz_{index}",
                        label_visibility="collapsed",
                    )
                submitted = st.form_submit_button("Submit Quiz")
            if submitted:
                score, details = score_quiz(st.session_state.quiz, answers)
                st.session_state.quiz_submitted = True
                st.success(f"Score: {score}/{len(st.session_state.quiz)}")
                for index, detail in enumerate(details, start=1):
                    status = "Correct" if detail["correct"] else "Needs review"
                    st.markdown(f"**{index}. {status}**")
                    st.write(f"Your answer: {detail['selected']}")
                    st.write(f"Correct answer: {detail['answer']}")
                    st.caption(detail["explanation"])

    with tab_flashcards:
        cards = generate_flashcards(full_text)
        if not cards:
            st.warning("No flashcards could be generated yet.")
        for index, card in enumerate(cards, start=1):
            with st.expander(f"Flashcard {index}: {card['question']}"):
                st.write(card["answer"])

    with tab_topics:
        topics = extract_topics(full_text)
        if topics:
            chip_html = "".join(f'<span class="topic-chip">{escape(topic)}</span>' for topic in topics)
            st.markdown(chip_html, unsafe_allow_html=True)
        else:
            st.warning("No topics found yet.")


def main() -> None:
    """Run the Streamlit app."""
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()

    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_NAME}</h1>
            <p class="muted">{APP_TAGLINE}</p>
            <p>Upload notes, papers, or study guides. Ask questions, inspect evidence,
            generate summaries, practice quizzes, and review flashcards.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sources = get_indexed_sources()
    chunk_count = sum(source["chunks"] for source in sources)
    workspace = escape(st.session_state.workspace_name)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-card"><b>Workspace</b><br>{workspace}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><b>Sources</b><br>{len(sources)}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><b>Chunks</b><br>{chunk_count}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        render_chat()
    with right:
        render_study_tools()


if __name__ == "__main__":
    main()
