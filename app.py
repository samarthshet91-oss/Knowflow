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
    background: radial-gradient(circle at top left, #0f172a 0%, #020617 45%, #000814 100%);
    color: #e2e8f0;
    font-family: "Segoe UI", sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    border-right: 1px solid rgba(148,163,184,0.15);
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(96,165,250,0.22);
    box-shadow: 0 0 30px rgba(59,130,246,0.12);
    margin-bottom: 1.5rem;
}

.hero h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2.5rem;
    color: #93c5fd;
}

.hero p {
    color: #cbd5e1;
    line-height: 1.7;
}

.metric-card {
    padding: 1rem;
    border-radius: 16px;
    background: linear-gradient(145deg, #0f172a, #1e293b);
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 0 14px rgba(59,130,246,0.08);
    text-align: center;
    min-height: 100px;
}

.topic-chip {
    display: inline-block;
    margin: 0.35rem;
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 0.9rem;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(96,165,250,0.2);
}

.mode-card {
    padding: 0.8rem 1rem;
    border-radius: 14px;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.12);
    margin-bottom: 0.8rem;
}

[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    padding: 0.6rem;
}

.stButton > button, button[kind="primary"] {
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    font-weight: 600 !important;
}

.stRadio > div {
    gap: 0.4rem;
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
        "app_mode": "Student",
        "quiz_difficulty": "Medium",
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
        st.image("assets/logo.jpeg",width=90)
        st.title("KnowFlow")
        st.caption("Your AI-Powered workspace for understanding , not just reading.")
        st.info("Local AI workspace powered by RAG + Ollama")
        st.text_input("Workspace name", key="workspace_name")
        st.markdown("## 🚀 Workspace Control")
        st.caption("Upload, manage, and explore your knowledge")
        # ---------------- MODE SELECTION ----------------
        st.markdown("### 🧠 Mode")

        mode = st.selectbox(
    "Choose role",
    ["Student", "Teacher", "Tutor", "Coder"],
           index=["Student", "Teacher", "Tutor", "Coder"].index(
             st.session_state.get("app_mode", "Student")
           )
       )

        st.session_state.app_mode = mode

# ---------------- QUIZ DIFFICULTY ----------------
        st.markdown("### 🎯 Quiz Difficulty")

        difficulty = st.selectbox(
    "Select difficulty",
    ["Easy", "Medium", "Hard"],
          index=["Easy", "Medium", "Hard"].index(
        st.session_state.get("quiz_difficulty", "Medium")
    )
        )
        st.session_state.quiz_difficulty = difficulty
        pages = ["✨ Welcome", "🏠 Home", "💬 Chat", "📑 Summary", "🧠 Flashcards", "❓ Quiz", "🔍 Topics"]
        if "current_page" not in st.session_state:
         st.session_state.current_page = "✨ Welcome"

        page = st.radio(
    "Go to",
    pages,
    index=pages.index(st.session_state.current_page)
)

        if st.session_state.current_page != page:
            st.session_state.current_page = page
            st.rerun()

        st.divider()
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
                st.write("Indexed file:", parsed.filename)
                st.write("Parsed text length:", len(parsed.text))

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
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared.")

            
def render_welcome_page() -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/logo.jpeg", width=150)

    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to KnowFlow</h1>
            <p class="muted">Source-grounded AI workspace for study and research.</p>
            <p>Upload notes, PDFs, and study guides. Chat with your sources, generate summaries, practice quizzes, review flashcards, and extract topics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🚀 Core Features")
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='metric-card'><b>Chat</b><br>Grounded answers</div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><b>Study Tools</b><br>Summary, quiz, flashcards</div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><b>Backend</b><br>FastAPI + Ollama</div>", unsafe_allow_html=True)            
            
def render_home() -> None:
    sources = get_indexed_sources()
    chunk_count = sum(source["chunks"] for source in sources)
    workspace = escape(st.session_state.get("workspace_name", "My Knowledge Workspace"))
    current_mode = st.session_state.get("app_mode", "Student")

    st.subheader("🏠 Dashboard")

    st.markdown(
        f"""
        <div class="hero">
            <h1>KnowFlow</h1>
            <p class="muted">Source-grounded AI workspace for study and research.</p>
            <p>Upload notes, PDFs, or study guides. Chat with your sources, inspect evidence, generate summaries, flashcards, quizzes, and key topics.</p>
            <p><b>Active mode:</b> {current_mode}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f'<div class="metric-card"><b>Workspace</b><br>{workspace}</div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-card"><b>Sources</b><br>{len(sources)}</div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-card"><b>Indexed Chunks</b><br>{chunk_count}</div>',
        unsafe_allow_html=True,
    )

def render_summary_page() -> None:
    import requests

    st.subheader("📄 Summary")
    st.write("Summary page loaded")
    st.divider()

    col1, col2, col3 = st.columns(3)

    mode = None
    if col1.button("Quick Summary", use_container_width=True):
        mode = "quick"
    if col2.button("Detailed Summary", use_container_width=True):
        mode = "detailed"
    if col3.button("Exam Summary", use_container_width=True):
        mode = "exam"

    if mode:
        with st.spinner("Generating summary..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/summary",
                    json={"mode": mode},
                    timeout=60,
                )
                data = response.json()
                st.session_state.summary = data.get("summary", "No summary returned from backend.")
            except Exception as e:
                st.error(f"Summary backend error: {e}")
                return

    st.markdown(st.session_state.get("summary", "Choose a summary style above."))


def render_chat() -> None:
    """Main source-grounded chat interface."""
    st.subheader("💬 Chat with your Sources")
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
        st.markdown(f"""
<div style="
    background-color:#0f1419;
    padding:12px;
    border-radius:10px;
    text-align:right;
    border:1px solid #2d3742;
">
{question}
</div>
""", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        try:
            model, model_name = load_embedding_model()
            st.session_state.embedding_status = model_name
            chunks = retrieve_relevant_chunks(question, model, DEFAULT_TOP_K)
            if not chunks:
                answer = "I could not find relevant evidence. Add sources or try a more specific question."
                mode = "No retrieval result"
            else:
                short_question = (
                    question
                    + "\n\nAnswer in a concise, user-friendly way."
                    + "Use 3 to 5 bullet points maximum."
                    + "Do not be overly long."
                )
                answer,mode = answer_question(short_question, chunks)   
            st.markdown(f"""
<div style="
    background-color:#1e1e1e;
    padding:15px;
    border-radius:12px;
    border-left:4px solid #4CAF50;
    margin-bottom:10px;
">
{answer}
</div>
""", unsafe_allow_html=True)
            st.caption(f"Generation mode: {mode}")

            if chunks:
                with st.expander("🔗Evidence and source chunks"):
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
            
def render_quiz_page() -> None:
    import requests

    st.subheader("❓ Quiz")
    st.divider()

    # number of questions
    num_questions = st.radio(
        "Number of questions",
        [5, 10],
        horizontal=True
    )

    # generate button
    if st.button("Generate Quiz", use_container_width=True):
        with st.spinner("Generating quiz..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/quiz",
                    json={"num_questions": num_questions},
                    timeout=60
                )
                data = response.json()

                st.session_state.quiz = data.get("quiz", [])
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False

            except Exception as e:
                st.error(f"Backend error: {e}")
                return

    quiz = st.session_state.get("quiz", [])

    # display quiz
    if quiz:
        st.write("### Answer the questions:")

        for i, q in enumerate(quiz):
            st.write(f"**Q{i+1}. {q['question']}**")

            selected = st.radio(
                f"Select answer for Q{i+1}",
                q["options"],
                key=f"q_{i}"
            )

            st.session_state.quiz_answers[i] = selected

        # submit button
        if st.button("Submit Quiz"):
            score = 0

            for i, q in enumerate(quiz):
                if st.session_state.quiz_answers.get(i) == q["answer"]:
                    score += 1

            st.success(f"Your Score: {score} / {len(quiz)}")  
def render_topics_page() -> None:
    st.subheader("🔍 Topics Explorer")
    st.caption("Extract key topics from your uploaded sources")

    st.divider()

    full_text = get_all_document_text(MAX_FULL_TEXT_CHARS)

    if not full_text:
        st.info("Upload a source or load the demo document to extract topics.")
        return

    # Generate topics button
    if st.button("Extract Topics", use_container_width=True):
        with st.spinner("Analyzing content and extracting topics..."):
            topics = extract_topics(full_text)
            st.session_state.topics = topics

    topics = st.session_state.get("topics", [])

    if not topics:
        st.warning("No topics generated yet. Click 'Extract Topics' to begin.")
        return

    # Display topics as premium chips
    st.markdown("### 🧠 Key Topics")

    chip_html = "".join(
        f"""
        <span style="
            display:inline-block;
            margin:6px;
            padding:8px 14px;
            border-radius:999px;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color:white;
            font-size:14px;
            font-weight:600;
            box-shadow: 0 0 12px rgba(96,165,250,0.3);
        ">
        {escape(topic)}
        </span>
        """
        for topic in topics
    )

    st.markdown(chip_html, unsafe_allow_html=True)

    st.divider()

    # Optional: topic details (advanced)
    selected_topic = st.selectbox("Explore a topic", topics)

    if selected_topic:
        st.markdown(f"### 📘 About: {selected_topic}")

        related_chunks = retrieve_relevant_chunks(selected_topic, load_embedding_model()[0])

        for chunk in related_chunks[:3]:
            st.markdown(f"""
            <div style="
                background:#0f172a;
                padding:10px;
                border-radius:10px;
                border:1px solid #334155;
                margin-bottom:10px;
            ">
            {chunk.text[:300]}...
            </div>
            """, unsafe_allow_html=True)                      
    
def render_flashcards_page() -> None:
    st.subheader("🧠 Flashcards")
    st.divider()
    full_text = get_all_document_text(MAX_FULL_TEXT_CHARS)
    if not full_text:
        st.info("Upload a source or load the demo document to unlock summaries, quizzes, topics, and flashcards.")
        return
    cards = generate_flashcards(full_text)
    if not cards:
        st.warning("No flashcards could be generated yet.")
        return
    for index, card in enumerate(cards, start=1):
        with st.expander(f"Flashcard {index}: {card['question']}"):
            st.write(card["answer"])    


def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()


page = st.session_state.get("current_page","✨ Welcome")
if page == "✨ Welcome":
    render_welcome_page()
elif page == "🏠 Home":
    render_home()
elif page == "💬 Chat":
    render_chat()
elif page == "📑 Summary":
    render_summary_page()
elif page == "🧠 Flashcards":
    render_flashcards_page()
elif page == "❓ Quiz":
    render_quiz_page()
elif page == "🔍 Topics":
    render_topics_page()


if __name__ == "__main__":
    main()
