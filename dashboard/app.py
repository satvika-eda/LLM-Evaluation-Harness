"""
Streamlit dashboard for the LLM Evaluation Harness.

Sections
--------
1. Leaderboard     — styled dataframe ranked by faithfulness, green=best/red=worst per column
2. Cost vs Quality — plotly scatter (cost vs faithfulness, bubble=BERTScore F1)
3. Metric Explorer — dropdown + bar chart with error bars across all models
4. Run New Eval    — form to submit a job, with live polling every 5 s
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

FASTAPI_URL: str = os.environ.get("FASTAPI_URL", "http://localhost:8000").rstrip("/")

KNOWN_MODELS: list[str] = [
    "llama-3.1-8b",
    "qwen2.5-72b",
    "deepseek-v3.2",
]

KNOWN_DATASETS: list[str] = ["truthfulqa", "hotpotqa"]
DATASET_LABELS: dict[str, str] = {
    "truthfulqa": "TruthfulQA",
    "hotpotqa": "HotpotQA",
}

ALL_METRICS: list[str] = [
    "ragas/faithfulness",
    "ragas/answer_relevance",
    "ragas/context_recall",
    "deepeval/hallucination",
    "deepeval/coherence",
    "bertscore/precision",
    "bertscore/recall",
    "bertscore/f1",
]

METRIC_LABELS: dict[str, str] = {
    "ragas/faithfulness": "Faithfulness (RAGAS)",
    "ragas/answer_relevance": "Answer Relevance (RAGAS)",
    "ragas/context_recall": "Context Recall (RAGAS)",
    "deepeval/hallucination": "Hallucination (DeepEval)",
    "deepeval/coherence": "Coherence (DeepEval)",
    "bertscore/precision": "BERTScore Precision",
    "bertscore/recall": "BERTScore Recall",
    "bertscore/f1": "BERTScore F1",
}

# Metrics where a lower score is better (colour scale is inverted)
LOWER_IS_BETTER: set[str] = {"deepeval/hallucination"}


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def fetch_leaderboard(dataset: str | None = None) -> list[dict]:
    params = {"dataset": dataset} if dataset else None
    resp = requests.get(f"{FASTAPI_URL}/leaderboard", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["leaderboard"]


@st.cache_data(ttl=30, show_spinner=False)
def fetch_runs() -> list[dict]:
    resp = requests.get(f"{FASTAPI_URL}/runs", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_runs_fresh() -> list[dict]:
    """Bypass cache — used during active job polling."""
    resp = requests.get(f"{FASTAPI_URL}/runs", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_results(run_id: int) -> dict:
    resp = requests.get(f"{FASTAPI_URL}/results/{run_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()["results"]


def fetch_aggregate_scores(dataset: str | None = None) -> dict[str, dict[str, list[float]]]:
    """
    Walk all completed runs and aggregate per-run metric averages.
    Returns {model: {metric: [avg_from_run_1, avg_from_run_2, ...]}}
    so the caller can compute cross-run mean and std-dev.

    When ``dataset`` is given, only runs on that dataset are included, so
    dataset-dependent metrics (faithfulness, context-recall) aren't blended
    across datasets that do and don't ship context.
    """
    runs = fetch_runs()
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run.get("status") != "completed":
            continue
        if dataset and run.get("dataset_name") != dataset:
            continue
        try:
            results = fetch_results(run["id"])
            for model, metrics in results.items():
                for metric, avg_score in metrics.items():
                    agg[model][metric].append(float(avg_score))
        except Exception:
            continue
    return agg


def post_run_eval(payload: dict) -> dict:
    resp = requests.post(f"{FASTAPI_URL}/run-eval", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Eval Harness",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🧪 LLM Evaluation Harness")
st.caption(f"Connected to backend: `{FASTAPI_URL}`")

# ── Dataset filter (applies to Leaderboard, Cost vs Quality, Metric Explorer) ──
# Faithfulness and context-recall are only meaningful on datasets that ship
# retrieval context (HotpotQA); scoping by dataset avoids blending them with a
# context-free dataset (TruthfulQA) where they read as ~0.
_DATASET_FILTER_OPTIONS = ["__all__", *KNOWN_DATASETS]
selected_dataset = st.selectbox(
    "Dataset filter",
    options=_DATASET_FILTER_OPTIONS,
    format_func=lambda d: "All datasets" if d == "__all__" else DATASET_LABELS.get(d, d),
    help=(
        "Scope the aggregate views to one dataset. Faithfulness and context "
        "recall are only meaningful on datasets with context (HotpotQA)."
    ),
    key="dataset_filter",
)
dataset_filter: str | None = None if selected_dataset == "__all__" else selected_dataset

# Pre-fetch leaderboard data (shared between tab 1 and tab 2)
leaderboard_data: list[dict] = []
leaderboard_error: str | None = None
try:
    leaderboard_data = fetch_leaderboard(dataset_filter)
except Exception as exc:
    leaderboard_error = str(exc)

tab_lb, tab_cv, tab_me, tab_run = st.tabs([
    "🏆 Leaderboard",
    "📊 Cost vs Quality",
    "🔍 Metric Explorer",
    "🚀 Run New Eval",
])

# ── Tab 1 — Leaderboard ───────────────────────────────────────────────────────

with tab_lb:
    st.subheader("Model Leaderboard")
    st.caption(
        "Ranked by average RAGAS Faithfulness (descending). "
        "🟢 = best · 🔴 = worst for each column. "
        "For Hallucination and Cost, lower values are better."
    )

    if leaderboard_error:
        st.error(f"Could not load leaderboard: {leaderboard_error}")
    elif not leaderboard_data:
        st.info("No completed eval runs yet. Head to **Run New Eval** to get started.")
    else:
        rows = []
        for entry in leaderboard_data:
            m = entry.get("metrics", {})
            rows.append(
                {
                    "Rank": entry["rank"],
                    "Model": entry["model"],
                    "Faithfulness": m.get("ragas/faithfulness"),
                    "Answer Relevance": m.get("ragas/answer_relevance"),
                    "Hallucination": m.get("deepeval/hallucination"),
                    "BERTScore F1": m.get("bertscore/f1"),
                    "Avg Cost / Q ($)": entry.get("avg_cost_per_question"),
                }
            )

        df_lb = pd.DataFrame(rows).set_index("Rank")

        # fmt: off
        fmt = {
            "Faithfulness":    "{:.3f}",
            "Answer Relevance":"{:.3f}",
            "Hallucination":   "{:.3f}",
            "BERTScore F1":    "{:.3f}",
            "Avg Cost / Q ($)":"${:.6f}",
        }
        # fmt: on

        # Columns where a lower value is better → reverse colour map
        col_cmap = {
            "Faithfulness":    "RdYlGn",
            "Answer Relevance":"RdYlGn",
            "Hallucination":   "RdYlGn_r",
            "BERTScore F1":    "RdYlGn",
            "Avg Cost / Q ($)":"RdYlGn_r",
        }

        styler = df_lb.style.format(fmt, na_rep="—")
        for col, cmap in col_cmap.items():
            if col in df_lb.columns and df_lb[col].notna().any():
                styler = styler.background_gradient(cmap=cmap, subset=[col], axis=0)

        st.dataframe(styler, use_container_width=True, height=min(400, 80 + 45 * len(rows)))

# ── Tab 2 — Cost vs Quality ───────────────────────────────────────────────────

with tab_cv:
    st.subheader("Cost vs Quality")
    st.caption(
        "X-axis = average cost per question (USD) · "
        "Y-axis = average faithfulness · "
        "Bubble size = BERTScore F1"
    )

    if leaderboard_error:
        st.error(f"Could not load data: {leaderboard_error}")
    elif not leaderboard_data:
        st.info("No data available. Run an evaluation first.")
    else:
        scatter_rows = []
        for entry in leaderboard_data:
            m = entry.get("metrics", {})
            cost = entry.get("avg_cost_per_question")
            faith = m.get("ragas/faithfulness")
            bf1 = m.get("bertscore/f1")
            if cost is not None and faith is not None:
                scatter_rows.append(
                    {
                        "Model": entry["model"],
                        "Avg Cost / Q ($)": cost,
                        "Faithfulness": faith,
                        # Guard against missing BERTScore — use mid-point as fallback
                        "BERTScore F1": bf1 if bf1 is not None else 0.5,
                    }
                )

        if not scatter_rows:
            st.info("Cost or faithfulness scores are not yet available.")
        else:
            df_sc = pd.DataFrame(scatter_rows)
            # BERTScore F1 with rescale_with_baseline can go negative for
            # dissimilar text; plotly marker size must be >= 0. Use a clamped
            # copy for the bubble size and keep the true F1 for the hover label.
            df_sc["_bubble"] = df_sc["BERTScore F1"].clip(lower=0.02)
            fig_sc = px.scatter(
                df_sc,
                x="Avg Cost / Q ($)",
                y="Faithfulness",
                size="_bubble",
                text="Model",
                color="Model",
                size_max=70,
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white",
                hover_data={
                    "Avg Cost / Q ($)": ":.6f",
                    "Faithfulness": ":.3f",
                    "BERTScore F1": ":.3f",
                    "_bubble": False,
                },
            )
            fig_sc.update_traces(
                textposition="top center",
                marker=dict(opacity=0.82, line=dict(width=1.5, color="white")),
            )
            fig_sc.update_layout(
                xaxis_title="Avg Cost per Question (USD)",
                yaxis_title="Avg Faithfulness Score",
                showlegend=False,
                height=500,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 3 — Metric Explorer ───────────────────────────────────────────────────

with tab_me:
    st.subheader("Metric Explorer")
    st.caption(
        "Select any metric to compare all models. "
        "Error bars show std deviation across completed eval runs."
    )

    selected_metric = st.selectbox(
        "Metric",
        options=ALL_METRICS,
        format_func=lambda k: METRIC_LABELS.get(k, k),
        key="metric_explorer_select",
    )

    agg_error: str | None = None
    agg_scores: dict[str, dict[str, list[float]]] = {}
    try:
        agg_scores = fetch_aggregate_scores(dataset_filter)
    except Exception as exc:
        agg_error = str(exc)

    if agg_error:
        st.error(f"Could not aggregate metric data: {agg_error}")
    elif not agg_scores:
        st.info("No completed runs found. Run an evaluation first.")
    else:
        bar_rows = []
        for model, metrics in agg_scores.items():
            scores = metrics.get(selected_metric, [])
            if scores:
                bar_rows.append(
                    {
                        "Model": model,
                        "Mean": float(np.mean(scores)),
                        "Std": float(np.std(scores, ddof=0)) if len(scores) > 1 else 0.0,
                        "Runs": len(scores),
                    }
                )

        if not bar_rows:
            label = METRIC_LABELS.get(selected_metric, selected_metric)
            st.info(f"No data available for **{label}** yet.")
        else:
            # Higher is better → sort descending; lower is better → sort ascending
            reverse_sort = selected_metric not in LOWER_IS_BETTER
            bar_rows.sort(key=lambda r: r["Mean"], reverse=reverse_sort)
            df_bar = pd.DataFrame(bar_rows)

            palette = px.colors.qualitative.Set2
            colours = [palette[i % len(palette)] for i in range(len(df_bar))]

            fig_bar = go.Figure(
                go.Bar(
                    x=df_bar["Model"],
                    y=df_bar["Mean"],
                    error_y=dict(
                        type="data",
                        array=df_bar["Std"].tolist(),
                        visible=True,
                        color="#444",
                        thickness=2,
                        width=8,
                    ),
                    marker_color=colours,
                    text=[f"{v:.3f}" for v in df_bar["Mean"]],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Mean: %{y:.4f}<br>"
                        "Std: %{error_y.array:.4f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

            direction = "↓ lower is better" if selected_metric in LOWER_IS_BETTER else "↑ higher is better"
            fig_bar.update_layout(
                title=dict(
                    text=f"{METRIC_LABELS.get(selected_metric, selected_metric)}  <sup>{direction}</sup>",
                    font=dict(size=16),
                ),
                yaxis=dict(title="Score", range=[0, 1.2], gridcolor="#eee"),
                xaxis_title="Model",
                template="plotly_white",
                height=460,
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("Raw statistics"):
                df_display = df_bar.rename(columns={"Mean": "Mean Score", "Std": "Std Dev", "Runs": "# Runs"})
                st.dataframe(
                    df_display.style.format(
                        {"Mean Score": "{:.4f}", "Std Dev": "{:.4f}", "# Runs": "{:.0f}"}
                    ),
                    use_container_width=True,
                )

# ── Tab 4 — Run New Eval ──────────────────────────────────────────────────────

with tab_run:
    st.subheader("Run a New Evaluation")

    # ── Active job polling banner ─────────────────────────────────────────────
    if "active_job" in st.session_state:
        active = st.session_state.active_job
        poll_container = st.container()
        with poll_container:
            st.markdown(
                f"**Active job** &nbsp;|&nbsp; "
                f"run_id: `{active['run_id']}` &nbsp; job_id: `{active['job_id']}`"
            )
            poll_placeholder = st.empty()
            try:
                fresh = fetch_runs_fresh()
                run_rec = next((r for r in fresh if r["id"] == active["run_id"]), None)

                if run_rec is None:
                    poll_placeholder.warning("Run record not found in database yet — retrying…")
                    time.sleep(5)
                    st.rerun()
                else:
                    status = run_rec["status"]
                    if status == "completed":
                        poll_placeholder.success("✅ Evaluation completed successfully!")
                        del st.session_state.active_job
                        # Invalidate caches so leaderboard/results update immediately
                        fetch_leaderboard.clear()
                        fetch_runs.clear()
                        fetch_results.clear()
                    elif status == "failed":
                        err_msg = run_rec.get("error_message") or "No details available."
                        poll_placeholder.error(f"❌ Evaluation failed:\n\n{err_msg}")
                        del st.session_state.active_job
                    else:
                        poll_placeholder.info(
                            f"⏳ Status: **{status}** — checking again in 5 seconds…"
                        )
                        time.sleep(5)
                        st.rerun()

            except Exception as poll_exc:
                poll_placeholder.warning(f"Could not poll status: {poll_exc}")

        st.divider()

    # ── Submission form ───────────────────────────────────────────────────────
    with st.form("run_eval_form", clear_on_submit=False):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            run_name = st.text_input(
                "Run name",
                value="my-eval-run",
                placeholder="e.g. baseline-truthfulqa-100",
                help="A short descriptive label stored with the run record.",
            )
            dataset = st.selectbox(
                "Dataset",
                options=KNOWN_DATASETS,
                format_func=lambda d: DATASET_LABELS.get(d, d),
                help="TruthfulQA: knowledge-only QA. HotpotQA: multi-hop QA with context.",
            )

        with col_right:
            n_questions = st.slider(
                "Number of questions",
                min_value=50,
                max_value=1000,
                value=100,
                step=50,
                help="Questions are sampled deterministically (seed=42) from the dataset.",
            )
            st.markdown("**Models to evaluate**")
            selected_models: list[str] = []
            for model in KNOWN_MODELS:
                if st.checkbox(model, value=True, key=f"chk_{model}"):
                    selected_models.append(model)

        st.markdown("")  # spacer
        submitted = st.form_submit_button(
            "🚀  Run Eval",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not run_name.strip():
            st.error("Please provide a run name.")
        elif not selected_models:
            st.error("Select at least one model to evaluate.")
        elif "active_job" in st.session_state:
            st.warning("A job is already running. Wait for it to finish before starting another.")
        else:
            with st.spinner("Submitting job to queue…"):
                try:
                    result = post_run_eval(
                        {
                            "run_name": run_name.strip(),
                            "dataset_name": dataset,
                            "n_questions": n_questions,
                            "models": selected_models,
                        }
                    )
                    st.session_state.active_job = {
                        "run_id": result["run_id"],
                        "job_id": result["job_id"],
                    }
                    st.rerun()
                except requests.HTTPError as http_exc:
                    try:
                        detail = http_exc.response.json().get("detail", http_exc.response.text)
                    except Exception:
                        detail = http_exc.response.text
                    st.error(f"API error {http_exc.response.status_code}: {detail}")
                except Exception as exc:
                    st.error(f"Failed to submit eval: {exc}")
