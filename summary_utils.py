"""Summary, topic, and flashcard helpers for KnowFlow."""

from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "between", "could",
    "during", "every", "first", "from", "have", "into", "more", "most", "other",
    "over", "should", "some", "such", "than", "that", "their", "there", "these",
    "this", "through", "using", "were", "when", "where", "which", "while", "with",
    "would", "your", "they", "them", "then", "what", "will", "been", "being",
}


def split_sentences(text: str) -> list[str]:
    """Split text into presentation-friendly sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]


def extract_topics(text: str, limit: int = 18) -> list[str]:
    """Extract important terms with a simple frequency-based method."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
    filtered = [word for word in words if word not in STOPWORDS]
    counts = Counter(filtered)
    topics = []
    for word, _count in counts.most_common(limit * 2):
        label = word.replace("_", " ").title()
        if label not in topics:
            topics.append(label)
        if len(topics) >= limit:
            break
    return topics


def rank_sentences(text: str, limit: int) -> list[str]:
    """Pick representative sentences using keyword overlap."""
    topics = {topic.lower() for topic in extract_topics(text, limit=30)}
    sentences = split_sentences(text)
    scored = []
    for index, sentence in enumerate(sentences):
        terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", sentence.lower()))
        score = len(terms & topics)
        scored.append((score, -index, sentence))
    scored.sort(reverse=True)
    selected = sorted(scored[:limit], key=lambda item: -item[1])
    return [sentence for _score, _index, sentence in selected]


def generate_summary(text: str, mode: str) -> str:
    """Generate a deterministic summary from uploaded content."""
    if not text.strip():
        return "No source text is available yet."

    mode = mode.lower()
    if mode == "quick":
        sentences = rank_sentences(text, 4)
        return "### Quick Summary\n" + "\n".join(f"- {sentence}" for sentence in sentences)

    if mode == "exam":
        topics = extract_topics(text, 10)
        sentences = rank_sentences(text, 6)
        topic_text = ", ".join(topics) if topics else "No clear topics found"
        return (
            "### Exam Summary\n"
            f"**High-yield topics:** {topic_text}\n\n"
            "**What to remember:**\n"
            + "\n".join(f"- {sentence}" for sentence in sentences)
            + "\n\n**Study tip:** Turn each bullet into a question and explain it aloud."
        )

    sentences = rank_sentences(text, 9)
    return (
        "### Detailed Summary\n"
        "This summary is generated only from the uploaded sources.\n\n"
        + "\n".join(f"- {sentence}" for sentence in sentences)
    )


def generate_flashcards(text: str, limit: int = 8) -> list[dict[str, str]]:
    """Create simple question-answer flashcards from important sentences."""
    cards = []
    topics = extract_topics(text, limit=limit)
    sentences = rank_sentences(text, limit=limit * 2)
    for topic in topics:
        topic_lower = topic.lower()
        answer = next((s for s in sentences if topic_lower.split()[0] in s.lower()), "")
        if answer:
            cards.append({"question": f"What should you know about {topic}?", "answer": answer})
        if len(cards) >= limit:
            break
    return cards
