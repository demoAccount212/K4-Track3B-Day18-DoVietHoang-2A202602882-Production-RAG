from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, GEMINI_API_KEY, USE_GEMINI, GEMINI_MODEL, OPENAI_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        # Try to get LLM for RAGAS
        llm = None
        if USE_GEMINI and GEMINI_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.1,
                )
            except Exception:
                pass  # Fall back to OpenAI or default
        
        if llm is None and OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=OPENAI_MODEL,
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0.1,
                )
            except Exception:
                pass

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        
        # Configure metrics to use our LLM if available
        if llm:
            result = evaluate(
                dataset, 
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm
            )
        else:
            # Let RAGAS use its default (requires OPENAI_API_KEY env var)
            result = evaluate(
                dataset, 
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
        df = result.to_pandas()
        per_question = [EvalResult(question=row["question"], answer=row["answer"],
            contexts=row["contexts"], ground_truth=row["ground_truth"],
            faithfulness=float(row.get("faithfulness", 0.0)),
            answer_relevancy=float(row.get("answer_relevancy", 0.0)),
            context_precision=float(row.get("context_precision", 0.0)),
            context_recall=float(row.get("context_recall", 0.0)))
            for _, row in df.iterrows()]

        aggregate = {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall": float(df["context_recall"].mean()),
        }
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def _to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    if not eval_results:
        return []

    # Compute average score for each question and find worst metric
    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": float(r.faithfulness),
            "answer_relevancy": float(r.answer_relevancy),
            "context_precision": float(r.context_precision),
            "context_recall": float(r.context_recall),
        }
        avg_score = sum(metrics.values()) / 4
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(worst_metric, ("Unknown", "Investigate"))

        scored.append({
            "question": r.question,
            "answer": r.answer,
            "contexts": r.contexts,
            "ground_truth": r.ground_truth,
            "avg_score": float(avg_score),
            "worst_metric": worst_metric,
            "worst_score": float(worst_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "metrics": metrics,
        })

    # Sort by avg_score ascending (worst first) and take bottom_n
    scored.sort(key=lambda x: x["avg_score"])
    return _to_serializable(scored[:bottom_n])


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: float(v) if hasattr(v, '__float__') else v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(report), f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
