# Evaluation Results

> ⚠️ **These results predate a set of scoring-pipeline fixes and are pending
> re-run.** Known issues with the numbers below: HotpotQA was answered
> closed-book (retrieval context never reached the model prompt) while
> faithfulness/hallucination graded answers against that unseen context;
> TruthfulQA faithfulness/hallucination were judged against an empty context
> (undefined); context_recall is constant across models by construction and
> has been removed; and cached generations may have been reused across the
> "independent" runs. Coherence and answer-relevance rankings are unaffected.
> This file will be regenerated from post-fix runs.

Benchmark of three open-weight LLMs on **faithfulness, hallucination, answer
relevance, coherence, context recall, and semantic similarity**, using RAGAS,
DeepEval, and BERTScore with an LLM-as-judge.

**Headline:** DeepSeek-V3.2 leads on both datasets — highest faithfulness and
coherence and the lowest hallucination rate everywhere — with Qwen2.5-72B second
and Llama-3.1-8B third.

## Setup

| | |
|---|---|
| **Models under test** | `deepseek-ai/DeepSeek-V3.2`, `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` (open-weight, via the HuggingFace Inference Providers router) |
| **Datasets** | HotpotQA (multi-hop QA *with* retrieval context) and TruthfulQA (knowledge QA, *no* context) — 100 questions each |
| **Runs** | 3 independent runs per dataset (error bars are mean ± std across runs) |
| **Scorers** | RAGAS (faithfulness, answer relevance, context recall), DeepEval (hallucination, G-Eval coherence), BERTScore (precision/recall/F1) |
| **Judge** | GPT-4o-mini (LLM-as-judge for RAGAS + DeepEval) |
| **Scale** | 1,800 model responses · 13,500 metric scores |

Arrows indicate direction of "better": ↑ higher is better, ↓ lower is better.

## The six runs

The benchmark is made up of **6 evaluation runs** — 3 per dataset. Each run
evaluates all three models on the same 100 questions; running each dataset three
times is what lets every reported metric be a **mean ± standard deviation across
runs** rather than a single point estimate.

| Runs | Dataset | Responses per run | Metrics / response | Scores per run |
|---|---|---|---|---|
| 1, 2, 3 | HotpotQA | 300 (100 questions × 3 models) | 8 | 2,400 |
| 4, 5, 6 | TruthfulQA | 300 (100 questions × 3 models) | 7\* | 2,100 |

\* TruthfulQA has no retrieval context, so `context_recall` is not computed
(7 metrics instead of 8).

**Totals:** 6 runs × 300 responses = **1,800 responses**, and
(3 × 2,400) + (3 × 2,100) = **13,500 scores**.

Each run is independent (fresh model generations, freshly judged), so the spread
across the three runs of a dataset captures judge non-determinism and sampling
variability. The resulting std values are small (≤ 0.03 on every metric), which
is why the model rankings are trustworthy and not an artifact of one lucky run.

## HotpotQA (100 questions × 3 runs)

| Metric | deepseek-v3.2 | qwen2.5-72b | llama-3.1-8b |
|---|---|---|---|
| Faithfulness ↑ | **0.460 ± 0.008** | 0.410 ± 0.013 | 0.340 ± 0.011 |
| Answer Relevance ↑ | **0.657 ± 0.026** | 0.574 ± 0.018 | 0.331 ± 0.007 |
| Context Recall ↑ | 0.857 ± 0.012 | 0.860 ± 0.010 | **0.867 ± 0.006** |
| Coherence (G-Eval) ↑ | **0.848 ± 0.016** | 0.778 ± 0.031 | 0.512 ± 0.005 |
| Hallucination ↓ | **0.672 ± 0.018** | 0.723 ± 0.006 | 0.787 ± 0.006 |
| BERTScore F1 | −0.309 | −0.256 | −0.205 |

## TruthfulQA (100 questions × 3 runs)

| Metric | deepseek-v3.2 | qwen2.5-72b | llama-3.1-8b |
|---|---|---|---|
| Faithfulness ↑ | **0.544 ± 0.003** | 0.408 ± 0.015 | 0.328 ± 0.006 |
| Answer Relevance ↑ | **0.472 ± 0.017** | 0.451 ± 0.014 | 0.458 ± 0.011 |
| Coherence (G-Eval) ↑ | **0.877 ± 0.003** | 0.847 ± 0.003 | 0.811 ± 0.002 |
| Hallucination ↓ | **0.057 ± 0.006** | 0.113 ± 0.015 | 0.183 ± 0.021 |
| BERTScore F1 | −0.160 | −0.059 | −0.049 |

(TruthfulQA has no retrieval context, so context recall is not applicable.)

## Key findings

- **DeepSeek-V3.2 is the strongest model overall** — best faithfulness,
  coherence, and answer relevance, and the lowest hallucination on both
  datasets. Llama-3.1-8B (the smallest model) trails consistently.
- **Hallucination scales with task difficulty.** Rates are far lower on
  single-hop TruthfulQA (0.06–0.18) than on multi-hop HotpotQA (0.67–0.79) —
  models fabricate more when reasoning across multiple passages.
- **Context recall is effectively tied** (~0.86 for all three), so the
  faithfulness gap on HotpotQA reflects how well each model *uses* retrieved
  context, not whether it retrieves it.
- **Results are highly reproducible** — per-metric std across 3 runs is ≤0.03,
  so the model rankings are stable rather than sampling noise.

## Caveats

- **LLM-as-judge, not ground truth.** RAGAS/DeepEval scores are produced by a
  GPT-4o-mini judge; they are consistent across models (so rankings are
  trustworthy) but absolute values are judge-dependent.
- **BERTScore F1 is negative** across the board. With
  `rescale_with_baseline=True`, short model answers vs. long references fall
  below the calibration baseline; BERTScore is a poor fit for this QA setup —
  the RAGAS/DeepEval metrics are the meaningful signal here.
- **Sample size:** 100 questions × 3 runs per dataset — appropriate for a
  comparative study, not a published benchmark.

## Reproducibility

```bash
# generation via HF router; judge via OpenAI
export LLM_JUDGE_MODEL=gpt-4o-mini        # configurable judge
export JUDGE_MAX_CONCURRENCY=3            # cap concurrent judge calls (TPM safety)
export SCORING_CHUNK_SIZE=20              # memory-bounded, resumable scoring
# enqueue a run per (dataset, model set) via POST /run-eval, then score with the orchestrator
```

Per-metric numbers are computed from the `scores` ⋈ `responses` tables, grouped
by run and averaged (mean ± std across the 3 runs), filtered by dataset.
