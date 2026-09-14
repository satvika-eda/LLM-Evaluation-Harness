"""
Custom DeepEval model wrapper for OpenAI-compatible judge endpoints.

DeepEval's built-in ``GPTModel`` hard-validates ``model`` against a fixed list
of real OpenAI model ids and, separately, never actually passes its
``base_url`` argument to the OpenAI client it builds (see
``deepeval.models.llms.openai_model.GPTModel.load_model``) — so pointing it at
an open-weight model id or a non-OpenAI ``base_url`` (e.g. the HF Inference
Providers router) fails or silently talks to api.openai.com anyway.

This wrapper implements DeepEval's ``DeepEvalBaseLLM`` interface directly
against any OpenAI-compatible chat-completions endpoint, so ``judge_config()``
(model / base_url / api_key) works for arbitrary judge models, not just GPT.
DeepEval treats any ``DeepEvalBaseLLM`` that isn't one of its native classes as
non-native and falls back to plain-text generation + JSON parsing (see
``deepeval.metrics.utils.initialize_model``), which this wrapper relies on by
not accepting a ``schema`` kwarg.
"""

from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM


class OpenAICompatibleJudgeModel(DeepEvalBaseLLM):
    """Judge model for any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str, api_key: str | None, base_url: str | None) -> None:
        self._model_id = model
        self._api_key = api_key
        self._base_url = base_url
        super().__init__(model_name=model)

    def load_model(self):
        from openai import AsyncOpenAI, OpenAI

        return {
            "sync": OpenAI(api_key=self._api_key, base_url=self._base_url),
            "async": AsyncOpenAI(api_key=self._api_key, base_url=self._base_url),
        }

    def generate(self, prompt: str) -> str:
        completion = self.model["sync"].chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        completion = await self.model["async"].chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return completion.choices[0].message.content

    def get_model_name(self) -> str:
        return self._model_id
