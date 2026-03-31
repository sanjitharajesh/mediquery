"""
backend/eval/ragas_eval.py

RAGAS evaluation for the MediQuery RAG pipeline.

Steps:
  1. Load eval_dataset.json (19 queries with expected_answer ground truths).
  2. Run every question through SimpleRAGChain to collect answer + contexts.
  3. Build a RAGAS Dataset with question / answer / contexts / ground_truth.
  4. Evaluate with faithfulness, answer_relevancy, context_precision.
  5. Print an aggregate results table plus a per-sample breakdown.
  6. Post per-sample RAGAS scores back to Langfuse, tied to each trace_id.
"""
import os
import sys
import json
import math
from pathlib import Path
from typing import List, Tuple

# Ensure backend/ is on sys.path regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision  # pre-instantiated Metric objects
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rag.chain import get_rag_chain, SimpleRAGChain
from observability.langfuse_tracing import langfuse_client

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 1. Build the RAGAS Dataset by running every query through the RAG chain
# ---------------------------------------------------------------------------

def build_ragas_dataset(
    chain: SimpleRAGChain,
) -> Tuple[Dataset, List[str]]:
    """
    Runs every question in eval_dataset.json through the RAG pipeline.

    Returns:
        dataset   — datasets.Dataset with question/answer/contexts/ground_truth
        trace_ids — parallel list of Langfuse trace IDs (one per query)
    """
    with open(EVAL_DATASET_PATH) as f:
        eval_data = json.load(f)

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []
    trace_ids: List[str] = []

    total = len(eval_data)
    print(f"\nRunning {total} queries through RAG pipeline...")
    print("-" * 65)

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        expected = item["expected_answer"]
        print(f"  [{i:2}/{total}] {question[:62]}")

        answer = chain.invoke(question)

        # _last_contexts and _last_trace_id are set inside invoke() after each call
        ctx = chain._last_contexts if chain._last_contexts else ["No context retrieved."]
        tid = chain._last_trace_id or ""

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(expected)
        trace_ids.append(tid)

    # Flush all pending Langfuse traces before proceeding
    langfuse_client.flush()
    print("-" * 65)
    print(f"Pipeline complete. {total} traces flushed to Langfuse.\n")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    return dataset, trace_ids


# ---------------------------------------------------------------------------
# 2. Run RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_eval(dataset: Dataset):
    """
    Configures RAGAS to use Groq (llama-3.3-70b-versatile) as the judge LLM
    and all-MiniLM-L6-v2 for answer_relevancy embeddings.

    Uses the pre-instantiated lowercase metric objects from ragas.metrics —
    these are the only objects that satisfy evaluate()'s isinstance(m, Metric)
    check. LLM and embeddings are injected via evaluate()'s own parameters,
    which handles the LangChain wrapper internally.
    """
    # Use llama-3.1-8b-instant for RAGAS judge calls to keep the 70b model's
    # token budget for the RAG pipeline itself (separate Groq per-model limits).
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    metrics = [faithfulness, answer_relevancy, context_precision]

    print("Running RAGAS evaluation (faithfulness / answer_relevancy / context_precision)...")
    print("This may take several minutes — each metric calls the LLM per sample.\n")

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=groq_llm,
        embeddings=hf_embeddings,
        run_config=RunConfig(max_workers=1),  # Groq only supports n=1; serialise calls
    )
    return result


# ---------------------------------------------------------------------------
# 3. Print results table
# ---------------------------------------------------------------------------

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision"]


def print_results_table(result) -> dict:
    """Prints aggregate scores and a per-sample breakdown."""

    result_df = result.to_pandas()

    # --- Aggregate table ---
    print("\n" + "=" * 56)
    print("  RAGAS EVALUATION — AGGREGATE SCORES")
    print("=" * 56)
    print(f"  {'Metric':<26} {'Score':>7}  {'Bar':}")
    print("  " + "-" * 48)

    scores = {}
    for name in METRIC_NAMES:
        if name in result_df.columns:
            score = result_df[name].mean()
            score = float("nan") if score != score else float(score)  # preserve NaN
        else:
            score = float("nan")
        scores[name] = score
        if math.isnan(score):
            bar = "n/a"
        else:
            filled = int(round(score * 20))
            bar = "█" * filled + "░" * (20 - filled)
        print(f"  {name:<26} {score:>7.4f}  {bar}")

    print("=" * 56)

    # --- Per-sample breakdown ---
    try:
        cols = ["question"] + [c for c in METRIC_NAMES if c in result_df.columns]
        df_display = result_df[cols].copy()
        df_display["question"] = df_display["question"].str[:45]
        print("\nPer-sample scores:")
        print(df_display.to_string(index=False))
    except Exception as exc:
        print(f"\n(Per-sample breakdown unavailable: {exc})")

    return scores


# ---------------------------------------------------------------------------
# 4. Post per-sample RAGAS scores back to Langfuse
# ---------------------------------------------------------------------------

def log_scores_to_langfuse(result, trace_ids: List[str]) -> None:
    """
    Posts per-sample faithfulness, answer_relevancy, and context_precision
    scores to Langfuse, each tied to the trace_id from the originating query.

    Scores with no trace_id (empty string) or NaN value are skipped silently.
    After all scores are posted, langfuse_client.flush() is called to ensure
    delivery before the process exits.
    """
    try:
        df = result.to_pandas()
    except Exception as exc:
        print(f"[Langfuse] Could not extract per-sample scores: {exc}")
        return

    posted = 0
    skipped = 0

    print("\nPosting RAGAS scores to Langfuse...")
    print("-" * 65)

    for i, (row, trace_id) in enumerate(zip(df.itertuples(index=False), trace_ids)):
        if not trace_id:
            print(f"  [{i+1:2}] SKIP  — no trace_id captured for this query")
            skipped += 1
            continue

        for metric_name in METRIC_NAMES:
            score_value = getattr(row, metric_name, None)
            if score_value is None or (isinstance(score_value, float) and math.isnan(score_value)):
                continue

            langfuse_client.create_score(
                trace_id=trace_id,
                name=metric_name,
                value=float(score_value),
                comment=f"RAGAS auto-eval — {metric_name}",
            )
            posted += 1

        question_preview = getattr(row, "question", "")[:55]
        print(
            f"  [{i+1:2}] trace={trace_id[:16]}…  "
            f"F={getattr(row, 'faithfulness', float('nan')):.3f}  "
            f"AR={getattr(row, 'answer_relevancy', float('nan')):.3f}  "
            f"CP={getattr(row, 'context_precision', float('nan')):.3f}  "
            f"| {question_preview}"
        )

    langfuse_client.flush()
    print("-" * 65)
    print(f"Done. {posted} scores posted, {skipped} rows skipped (no trace_id).\n")


# ---------------------------------------------------------------------------
# 5. Drift detection
# ---------------------------------------------------------------------------

DRIFT_THRESHOLD = 0.75
DRIFT_WINDOW = 5  # number of recent runs used for the rolling average


def check_faithfulness_drift(
    scores: List[float],
    threshold: float = DRIFT_THRESHOLD,
    window: int = DRIFT_WINDOW,
) -> dict:
    """
    Detects faithfulness degradation over recent eval runs.

    Takes a list of per-run aggregate faithfulness scores (one float per run,
    ordered oldest → newest) and computes a rolling average over the last
    `window` runs. Emits a printed alert and returns a result dict when the
    rolling average drops below `threshold`.

    Args:
        scores:    List of faithfulness scores from successive eval runs.
                   Pass a single-element list when checking after one run.
        threshold: Alert level. Default 0.75.
        window:    Rolling window size. Default 5.

    Returns:
        {
            "rolling_avg": float,
            "window_used": int,       # actual samples averaged (≤ window)
            "threshold": float,
            "drift_detected": bool,
            "scores_used": List[float],
        }
    """
    if not scores:
        raise ValueError("scores list is empty — nothing to evaluate.")

    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        raise ValueError("All scores are NaN — cannot compute rolling average.")

    window_scores = valid[-window:]
    rolling_avg = sum(window_scores) / len(window_scores)
    drift_detected = rolling_avg < threshold

    result = {
        "rolling_avg": round(rolling_avg, 4),
        "window_used": len(window_scores),
        "threshold": threshold,
        "drift_detected": drift_detected,
        "scores_used": window_scores,
    }

    print("\n" + "=" * 56)
    print("  FAITHFULNESS DRIFT CHECK")
    print("=" * 56)
    print(f"  Scores in window : {[round(s, 4) for s in window_scores]}")
    print(f"  Rolling avg      : {rolling_avg:.4f}  (window={len(window_scores)})")
    print(f"  Threshold        : {threshold}")

    if drift_detected:
        print(f"  Status           : *** DRIFT ALERT ***")
        print(f"  Rolling avg {rolling_avg:.4f} is below threshold {threshold}.")
        print(f"  Investigate retrieval quality or prompt changes.")
    else:
        print(f"  Status           : OK — no drift detected")

    print("=" * 56)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chain = get_rag_chain()
    dataset, trace_ids = build_ragas_dataset(chain)
    result = run_ragas_eval(dataset)
    scores = print_results_table(result)
    log_scores_to_langfuse(result, trace_ids)

    # Drift check — seed with just this run's faithfulness score.
    # In production, load the historical scores list from a store and append.
    current_faithfulness = scores.get("faithfulness", float("nan"))
    check_faithfulness_drift([current_faithfulness])

    print("Evaluation complete.")
