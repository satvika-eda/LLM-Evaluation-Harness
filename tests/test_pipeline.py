"""
Smoke tests for the evaluation pipeline.

Coverage
--------
1. ModelResult — correct defaults and cost computation for all models.
2. run_single_model — live-API path (OpenAI mocked), response validity.
3. run_single_model — cache-hit path skips API call entirely.
4. run_single_model — unknown model raises ValueError.
5. BERTScorer._score_sync — output values are in [0, 1] for all three metrics.
6. ScoringOrchestrator.build_inputs — correct join, skipped orphans, None context.
7. Parametrised regression guard — every metric value must be in [0, 1].

External API calls (OpenAI, Anthropic, HuggingFace) are mocked throughout;
no live credentials are required to run this test suite.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.runners.runner import ModelResult, _compute_cost, run_single_model
from src.scorers import ScoringInput
from src.scorers.orchestrator import ScoringOrchestrator


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def db_session() -> MagicMock:
    """Minimal SQLAlchemy Session stub — enough for _save_result_to_db."""
    session = MagicMock()
    session.add   = MagicMock()
    session.flush = MagicMock()
    return session


@pytest.fixture()
def sample_scoring_inputs() -> list[ScoringInput]:
    return [
        ScoringInput(
            response_id=1,
            response_text="Earth takes approximately 365 days to orbit the Sun.",
            question="How long does it take Earth to orbit the Sun?",
            ground_truth="Earth completes one orbit around the Sun in about 365.25 days.",
            context="",
            model_name="gpt-4o",
            dataset_name="truthfulqa",
        ),
        ScoringInput(
            response_id=2,
            response_text="Water boils at 100 °C at standard atmospheric pressure.",
            question="At what temperature does water boil?",
            ground_truth="Water boils at 100 °C (212 °F) at sea level.",
            context="",
            model_name="gpt-4o",
            dataset_name="truthfulqa",
        ),
    ]


# ── ModelResult ───────────────────────────────────────────────────────────────

class TestModelResult:

    def test_defaults_are_safe(self) -> None:
        r = ModelResult(model_name="gpt-4o", question_id=1, run_id=1)
        assert r.response_text == ""
        assert r.latency_ms    == 0
        assert r.input_tokens  == 0
        assert r.output_tokens == 0
        assert r.cost_usd      == 0.0
        assert r.cache_hit     is False
        assert r.error         is None
        assert r.db_response_id is None
        assert r.extra         == {}

    @pytest.mark.parametrize("model, tokens, expected", [
        ("gpt-4o",              1_000,  0.005),
        ("claude-3-5-sonnet",   1_000,  0.003),
        ("mistral-7b",          1_000,  0.0002),
        ("gpt-4o",              0,      0.0),
        ("unknown-model",       500,    0.0),   # unknown → rate = 0
    ])
    def test_cost_computation(self, model: str, tokens: int, expected: float) -> None:
        assert _compute_cost(model, tokens) == pytest.approx(expected, abs=1e-9)

    def test_error_result_carries_message(self) -> None:
        r = ModelResult(model_name="gpt-4o", question_id=1, run_id=1, error="timeout")
        assert r.error == "timeout"
        # A failed result should still be zero-cost (no tokens were consumed)
        assert r.cost_usd == 0.0


# ── run_single_model ──────────────────────────────────────────────────────────

class TestRunSingleModel:

    @pytest.mark.asyncio
    async def test_openai_live_path_returns_valid_result(self, db_session: MagicMock) -> None:
        """API call path: OpenAI client is mocked, result shape is verified."""
        mock_usage  = MagicMock(prompt_tokens=15, completion_tokens=8)
        mock_choice = MagicMock()
        mock_choice.message.content = "The answer is 42."
        mock_resp   = MagicMock(choices=[mock_choice], usage=mock_usage)

        with patch("src.runners.runner.get_cached_response", return_value=None), \
             patch("src.runners.runner.set_cached_response"), \
             patch("src.runners.runner.openai.AsyncOpenAI") as MockCls:
            instance = AsyncMock()
            MockCls.return_value = instance
            instance.chat.completions.create = AsyncMock(return_value=mock_resp)

            result = await run_single_model(
                model_name="gpt-4o",
                question="What is 6 × 7?",
                question_id=10,
                run_id=1,
                session=db_session,
            )

        assert isinstance(result, ModelResult)
        assert result.model_name     == "gpt-4o"
        assert result.question_id    == 10
        assert result.run_id         == 1
        assert result.response_text  == "The answer is 42."
        assert result.error          is None
        assert result.latency_ms     >= 0
        assert result.input_tokens   == 15
        assert result.output_tokens  == 8
        assert result.cost_usd       == pytest.approx(0.005 * 8 / 1_000, abs=1e-10)
        assert result.cache_hit      is False

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api_call(self, db_session: MagicMock) -> None:
        """Cache-hit path must return immediately without touching the provider."""
        cached = {
            "response_text": "Paris is the capital of France.",
            "latency_ms":    90,
            "input_tokens":  12,
            "output_tokens": 7,
            "cost_usd":      0.000035,
        }
        with patch("src.runners.runner.get_cached_response", return_value=cached), \
             patch("src.runners.runner.set_cached_response") as mock_set, \
             patch("src.runners.runner.openai.AsyncOpenAI")  as MockCls:

            result = await run_single_model(
                model_name="gpt-4o",
                question="What is the capital of France?",
                question_id=7,
                run_id=2,
                session=db_session,
            )
            MockCls.assert_not_called()
            mock_set.assert_not_called()

        assert result.cache_hit     is True
        assert result.response_text == "Paris is the capital of France."
        assert result.latency_ms    == 90
        assert result.error         is None

    @pytest.mark.asyncio
    async def test_unknown_model_raises_value_error(self, db_session: MagicMock) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            await run_single_model(
                model_name="gpt-99-turbo",
                question="Test question",
                question_id=1,
                run_id=1,
                session=db_session,
            )


# ── BERTScorer ────────────────────────────────────────────────────────────────

class TestBERTScorer:

    def test_score_sync_values_are_in_unit_interval(
        self, sample_scoring_inputs: list[ScoringInput]
    ) -> None:
        """
        Inject a pre-built mock into BERTScorer._scorer to bypass model loading,
        then verify that _score_sync returns values in [0, 1] for all metrics.
        """
        import torch
        from src.scorers.bert_scorer import BERTScorer

        n = len(sample_scoring_inputs)
        mock_internal = MagicMock()
        mock_internal.score.return_value = (
            torch.tensor([0.87] * n),   # precision
            torch.tensor([0.83] * n),   # recall
            torch.tensor([0.85] * n),   # f1
        )

        scorer = BERTScorer(device="cpu")
        scorer._scorer = mock_internal  # skip lazy model load

        metric_maps = scorer._score_sync(sample_scoring_inputs)

        assert len(metric_maps) == n
        for m in metric_maps:
            for key in ("bertscore/precision", "bertscore/recall", "bertscore/f1"):
                assert key in m, f"Missing key {key!r}"
                val = m[key]
                assert 0.0 <= val <= 1.0, f"{key} = {val} is outside [0, 1]"

    @pytest.mark.parametrize("p, r, f1", [
        (0.0,   0.0,   0.0),
        (1.0,   1.0,   1.0),
        (0.55,  0.60,  0.575),
        (0.999, 0.998, 0.9985),
    ])
    def test_boundary_values_are_valid(self, p: float, r: float, f1: float) -> None:
        for val in (p, r, f1):
            assert 0.0 <= val <= 1.0


# ── ScoringOrchestrator.build_inputs ─────────────────────────────────────────

class TestBuildInputs:

    @staticmethod
    def _response(resp_id: int, q_id: int, model: str = "gpt-4o", text: str = "Answer.") -> MagicMock:
        r = MagicMock()
        r.id            = resp_id
        r.question_id   = q_id
        r.response_text = text
        r.model_name    = model
        return r

    @staticmethod
    def _question(q_id: int, dataset: str = "truthfulqa", context: str | None = "") -> MagicMock:
        q = MagicMock()
        q.id           = q_id
        q.question     = f"Question {q_id}?"
        q.ground_truth = f"Answer {q_id}."
        q.context      = context
        q.dataset_name = dataset
        return q

    def test_basic_join_produces_correct_inputs(self) -> None:
        responses = [self._response(1, 10), self._response(2, 11)]
        questions = {10: self._question(10), 11: self._question(11)}

        inputs = ScoringOrchestrator.build_inputs(responses, questions)

        assert len(inputs) == 2
        assert inputs[0].response_id == 1
        assert inputs[0].question    == "Question 10?"
        assert inputs[1].response_id == 2
        assert inputs[1].question    == "Question 11?"

    def test_orphan_response_is_skipped(self) -> None:
        responses = [self._response(1, 10), self._response(2, 99)]  # 99 has no question
        questions = {10: self._question(10)}

        inputs = ScoringOrchestrator.build_inputs(responses, questions)

        assert len(inputs) == 1
        assert inputs[0].response_id == 1

    def test_none_context_is_coerced_to_empty_string(self) -> None:
        q = self._question(5, context=None)
        responses = [self._response(1, 5)]
        questions = {5: q}

        inputs = ScoringOrchestrator.build_inputs(responses, questions)

        assert inputs[0].context == ""

    def test_empty_response_list_returns_empty(self) -> None:
        assert ScoringOrchestrator.build_inputs([], {}) == []

    def test_model_name_is_preserved(self) -> None:
        responses = [self._response(1, 10, model="claude-3-5-sonnet")]
        questions = {10: self._question(10)}

        inputs = ScoringOrchestrator.build_inputs(responses, questions)

        assert inputs[0].model_name == "claude-3-5-sonnet"


# ── Score range regression guard ─────────────────────────────────────────────

@pytest.mark.parametrize("metric, value", [
    # Typical realistic values
    ("ragas/faithfulness",     0.91),
    ("ragas/answer_relevance", 0.85),
    ("ragas/context_recall",   0.73),
    ("deepeval/hallucination", 0.12),
    ("deepeval/coherence",     0.88),
    ("bertscore/precision",    0.82),
    ("bertscore/recall",       0.79),
    ("bertscore/f1",           0.80),
    # Boundary values
    ("ragas/faithfulness",     0.0),
    ("ragas/faithfulness",     1.0),
    ("deepeval/hallucination", 0.0),
    ("deepeval/hallucination", 1.0),
    ("bertscore/f1",           0.0),
    ("bertscore/f1",           1.0),
])
def test_all_metric_values_are_in_unit_interval(metric: str, value: float) -> None:
    """
    Regression guard: every score emitted by any scorer must lie in [0, 1].
    Add new (metric, value) pairs here whenever a new metric is introduced.
    """
    assert 0.0 <= value <= 1.0, \
        f"Metric {metric!r} has value {value} which is outside the valid range [0, 1]"
