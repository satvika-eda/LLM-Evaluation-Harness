"""
Database connection setup and SQLAlchemy ORM models for the LLM evaluation harness.

Tables
------
eval_runs   — top-level record for a single evaluation run
questions   — dataset questions with ground-truth answers and context
responses   — per-model responses generated during a run
scores      — individual metric scores attached to a response
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import enum

from dotenv import load_dotenv
from sqlalchemy import (
    ARRAY,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

load_dotenv()

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # recycle stale connections
    pool_size=5,
    max_overflow=10,
    echo=False,           # set True to log all SQL
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """Dependency-injectable session generator (FastAPI / plain context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RunStatus(str, enum.Enum):
    """Lifecycle states of an EvalRun, stored as a native PostgreSQL ENUM."""
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EvalRun(Base):
    """
    One top-level evaluation run.

    models_evaluated stores the list of model identifiers that were tested,
    e.g. ["gpt-4o", "claude-3-5-sonnet", "mistral-7b"].
    """

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # PostgreSQL native array; stores model IDs as text[]
    models_evaluated: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    status: Mapped[str] = mapped_column(
        SAEnum(
            RunStatus,
            name="run_status",
            create_type=True,
            # Store the lowercase enum values ("pending"), not the member
            # names ("PENDING"), so raw SQL like status = 'pending' works.
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=RunStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    responses: Mapped[list["Response"]] = relationship(
        "Response", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<EvalRun id={self.id} name={self.run_name!r} "
            f"status={self.status!r} dataset={self.dataset_name!r}>"
        )


class Question(Base):
    """
    A single question from a dataset, with its ground-truth answer and context.

    context holds the reference passage(s) used for RAG / faithfulness evaluation.
    It is nullable so the model can be used for non-RAG datasets too.
    """

    __tablename__ = "questions"
    __table_args__ = (
        UniqueConstraint("dataset_name", "question", name="uq_question_dataset_text"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)

    responses: Mapped[list["Response"]] = relationship(
        "Response", back_populates="question", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        preview = self.question[:60] + "..." if len(self.question) > 60 else self.question
        return f"<Question id={self.id} dataset={self.dataset_name!r} q={preview!r}>"


class Response(Base):
    """
    A model's raw response to one question within one eval run.

    Stores runtime telemetry (latency, token counts, cost) alongside the text
    so aggregate performance statistics can be queried directly from the DB.

    Failed calls are persisted too (response_text="" and error set) so a
    model's completion rate is queryable — otherwise a flaky model would
    simply have fewer rows and identical-looking averages.
    """

    __tablename__ = "responses"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "question_id", "model_name", name="uq_response_run_question_model"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(nullable=False, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="responses")
    question: Mapped["Question"] = relationship("Question", back_populates="responses")
    scores: Mapped[list["Score"]] = relationship(
        "Score", back_populates="response", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Response id={self.id} model={self.model_name!r} "
            f"run_id={self.run_id} question_id={self.question_id}>"
        )


class Score(Base):
    """
    A single metric score for one model response.

    One Response can have many Scores — one per metric
    (e.g. faithfulness=0.92, answer_relevance=0.87, bertscore_f1=0.81).
    """

    __tablename__ = "scores"
    __table_args__ = (
        UniqueConstraint("response_id", "metric_name", name="uq_score_response_metric"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    response: Mapped["Response"] = relationship("Response", back_populates="scores")

    def __repr__(self) -> str:
        return f"<Score id={self.id} metric={self.metric_name!r} score={self.score:.4f}>"


# ---------------------------------------------------------------------------
# Schema management helpers
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all tables that do not yet exist. Safe to call on startup."""
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all tables. Destructive — only use in tests or local resets."""
    Base.metadata.drop_all(bind=engine)
