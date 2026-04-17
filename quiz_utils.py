"""Deterministic quiz generation for KnowFlow."""

from __future__ import annotations

import random
import re

from summary_utils import extract_topics, rank_sentences


def _shorten(text: str, max_len: int = 170) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[: max_len - 3] + "..." if len(text) > max_len else text


def generate_quiz(text: str, question_count: int = 5) -> list[dict]:
    """Generate multiple-choice questions from document sentences.

    The questions ask users to match a source statement to its closest concept.
    This keeps the quiz reliable without needing a paid generation API.
    """
    if not text.strip():
        return []

    question_count = 10 if question_count == 10 else 5
    topics = extract_topics(text, limit=max(12, question_count + 4))
    sentences = rank_sentences(text, limit=max(20, question_count * 3))
    if len(topics) < 4 or not sentences:
        return []

    random.seed(42)
    quiz = []
    used_topics: set[str] = set()

    for topic in topics:
        if topic in used_topics:
            continue
        topic_word = topic.lower().split()[0]
        statement = next((s for s in sentences if topic_word in s.lower()), "")
        if not statement:
            continue

        distractors = [candidate for candidate in topics if candidate != topic]
        random.shuffle(distractors)
        options = distractors[:3] + [topic]
        random.shuffle(options)

        quiz.append(
            {
                "question": (
                    "Which concept is most closely connected to this source statement?\n\n"
                    f"'{_shorten(statement)}'"
                ),
                "options": options,
                "answer": topic,
                "explanation": _shorten(statement, 240),
            }
        )
        used_topics.add(topic)
        if len(quiz) >= question_count:
            break

    return quiz


def score_quiz(quiz: list[dict], user_answers: dict[int, str]) -> tuple[int, list[dict]]:
    """Score submitted quiz answers."""
    score = 0
    details = []
    for index, item in enumerate(quiz):
        selected = user_answers.get(index)
        correct = selected == item["answer"]
        if correct:
            score += 1
        details.append(
            {
                "question": item["question"],
                "selected": selected or "No answer",
                "answer": item["answer"],
                "correct": correct,
                "explanation": item["explanation"],
            }
        )
    return score, details
