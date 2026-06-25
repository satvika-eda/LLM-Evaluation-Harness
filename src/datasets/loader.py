"""
Dataset loader for the LLM evaluation harness.

Supports
--------
- TruthfulQA  (truthful_qa / multiple_choice) — validation split
- HotpotQA    (hotpot_qa  / distractor)       — validation split

Each loader returns a list of plain dicts with the canonical keys:
    question    : str   — the question text
    ground_truth: str   — correct answer text
    context     : str   — supporting passage(s); empty string when not applicable
    dataset_name: str   — identifier used in the DB

Public API
----------
    load_truthfulqa(split, n)           -> list[dict]
    load_hotpotqa(split, n)             -> list[dict]
    sample_questions(questions, n, seed) -> list[dict]
    save_questions_to_db(questions, session) -> list[Question]
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from datasets import load_dataset  # type: ignore[import-untyped]
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from datasets import Dataset  # type: ignore[import-untyped]

from src.db import Question

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TRUTHFULQA_NAME = "truthfulqa"
HOTPOTQA_NAME = "hotpotqa"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_truthfulqa_row(row: dict) -> dict:
    """
    Extract the correct answer from mc1_targets.

    mc1_targets has exactly one label==1; its corresponding choice is the
    ground-truth answer we use for evaluation.
    """
    choices: list[str] = row["mc1_targets"]["choices"]
    labels: list[int] = row["mc1_targets"]["labels"]

    # Guaranteed to find exactly one; fall back to first choice if malformed.
    ground_truth = next(
        (c for c, l in zip(choices, labels) if l == 1),
        choices[0],
    )

    return {
        "dataset_name": TRUTHFULQA_NAME,
        "question": row["question"].strip(),
        "ground_truth": ground_truth.strip(),
        "context": "",  # TruthfulQA is knowledge-only; no retrieval context
    }


def _parse_hotpotqa_row(row: dict) -> dict:
    """
    Build a single context string from all supporting paragraphs.

    HotpotQA context structure:
        {"title": [str, ...], "sentences": [[str, ...], ...]}

    Each element of `sentences` is the list of sentences for the corresponding
    title. We concatenate everything into one passage so scorers (RAGAS,
    BERTScore) receive a single string rather than a nested list.
    """
    titles: list[str] = row["context"]["title"]
    sentence_lists: list[list[str]] = row["context"]["sentences"]

    paragraphs: list[str] = []
    for title, sentences in zip(titles, sentence_lists):
        body = " ".join(s.strip() for s in sentences)
        paragraphs.append(f"{title}: {body}")

    context = "\n\n".join(paragraphs)

    return {
        "dataset_name": HOTPOTQA_NAME,
        "question": row["question"].strip(),
        "ground_truth": row["answer"].strip(),
        "context": context,
    }


# ── Public loaders ────────────────────────────────────────────────────────────

def load_truthfulqa(split: str = "validation", n: int | None = None) -> list[dict]:
    """
    Load TruthfulQA in multiple-choice format.

    Parameters
    ----------
    split:
        HuggingFace split name. Only "validation" exists for this config.
    n:
        If given, return at most n questions (taken from the start of the
        split, before any sampling). Use sample_questions() for random subsets.

    Returns
    -------
    list[dict] with keys: dataset_name, question, ground_truth, context
    """
    logger.info("Loading TruthfulQA (%s split)…", split)
    ds: Dataset = load_dataset("truthful_qa", "multiple_choice", split=split, trust_remote_code=False)

    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    questions = [_parse_truthfulqa_row(row) for row in ds]
    logger.info("Loaded %d TruthfulQA questions.", len(questions))
    return questions


def load_hotpotqa(split: str = "validation", n: int | None = None) -> list[dict]:
    """
    Load HotpotQA in the distractor setting.

    Parameters
    ----------
    split:
        HuggingFace split name — "train", "validation", or "test".
        Defaults to "validation" (7 410 rows) to avoid pulling 90k rows.
    n:
        If given, return at most n questions from the head of the split.

    Returns
    -------
    list[dict] with keys: dataset_name, question, ground_truth, context
    """
    logger.info("Loading HotpotQA distractor (%s split)…", split)
    ds: Dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=False)

    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    questions = [_parse_hotpotqa_row(row) for row in ds]
    logger.info("Loaded %d HotpotQA questions.", len(questions))
    return questions


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_questions(
    questions: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """
    Return a random sample of n questions from an already-loaded list.

    Parameters
    ----------
    questions:
        Output of load_truthfulqa() or load_hotpotqa() (or any concatenation
        of the two).
    n:
        Number of questions to sample. Clamped to len(questions) so it never
        raises.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    A new list of at most n dicts.
    """
    n = min(n, len(questions))
    rng = random.Random(seed)
    sampled = rng.sample(questions, n)
    logger.debug("Sampled %d / %d questions (seed=%d).", n, len(questions), seed)
    return sampled


# ── Persistence ───────────────────────────────────────────────────────────────

def save_questions_to_db(
    questions: list[dict],
    session: Session,
    *,
    skip_duplicates: bool = True,
) -> list[Question]:
    """
    Persist a list of question dicts to the `questions` table.

    Parameters
    ----------
    questions:
        Each dict must contain: dataset_name, question, ground_truth, context.
    session:
        An active SQLAlchemy Session. The caller is responsible for
        commit / rollback so this function can participate in larger
        transactions.
    skip_duplicates:
        When True (default), questions whose (dataset_name, question) pair
        already exists in the DB are silently skipped. When False, they are
        inserted again (useful for tests with isolated transactions).

    Returns
    -------
    list[Question] — the ORM objects that were actually added to the session
    (does not include skipped duplicates).
    """
    if not questions:
        return []

    added: list[Question] = []

    # Build a set of existing (dataset_name, question) pairs in one query
    # so we don't hit the DB once per row in the loop.
    existing: set[tuple[str, str]] = set()
    if skip_duplicates:
        dataset_names = {q["dataset_name"] for q in questions}
        rows = (
            session.query(Question.dataset_name, Question.question)
            .filter(Question.dataset_name.in_(dataset_names))
            .all()
        )
        existing = {(r.dataset_name, r.question) for r in rows}

    for q in questions:
        key = (q["dataset_name"], q["question"])
        if skip_duplicates and key in existing:
            continue

        obj = Question(
            dataset_name=q["dataset_name"],
            question=q["question"],
            ground_truth=q["ground_truth"],
            context=q["context"] or None,  # store empty string as NULL
        )
        session.add(obj)
        added.append(obj)

    session.flush()  # assign IDs without committing the transaction
    logger.info(
        "Saved %d questions to DB (%d skipped as duplicates).",
        len(added),
        len(questions) - len(added),
    )
    return added
