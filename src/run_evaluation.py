"""
Step 4 — Run all 20 evaluation questions through the RAG pipeline.

Run from project root:
    python src/run_evaluation.py

Reads:  outputs/eval_questions.csv
Writes: outputs/eval_results.csv
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rag_query import get_rag_answer

CATEGORY_LABELS = {
    "factual":      "Clearly answerable from documents",
    "partial":      "Partially answerable",
    "out_of_scope": "Not answerable — should refuse",
    "adversarial":  "Adversarial / hallucination bait",
}


def run_evaluation(
    questions_file: str = "outputs/eval_questions.csv",
    results_file: str = "outputs/eval_results.csv",
    vectorstore_dir: str = "vectorstore",
    embedding_model: str = "all-MiniLM-L6-v2",
    delay_seconds: float = 1.0,
):
    """
    Run every question in the eval set through the RAG pipeline.

    Args:
        questions_file:  Path to the evaluation CSV
        results_file:    Where to save results
        vectorstore_dir: ChromaDB directory
        embedding_model: Must match build_vectorstore.py
        delay_seconds:   Pause between API calls (avoids rate limits)
    """
    df = pd.read_csv(questions_file)
    print(f"📋  Loaded {len(df)} evaluation questions\n")
    print("Question categories:")
    for cat, label in CATEGORY_LABELS.items():
        n = len(df[df["category"] == cat])
        print(f"    • {cat:<15} {n} questions  ({label})")
    print()

    results = []
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌  GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key'")
        sys.exit(1)

    for idx, row in df.iterrows():
        question = row["question"]
        category = row["category"]
        expected = row["expected_behavior"]

        print(f"[{idx+1:02d}/{len(df)}] [{category}] {question[:70]}...")

        try:
            result = get_rag_answer(
                question,
                vectorstore_dir=vectorstore_dir,
                embedding_model=embedding_model,
                api_key=api_key,
            )

            # Determine if refusal was correct
            should_refuse = category == "out_of_scope"
            did_refuse = result["refused"]
            refusal_correct = (should_refuse == did_refuse)

            chunks = result["retrieved_chunks"]

            results.append({
                "question":         question,
                "category":         category,
                "expected_behavior": expected,
                "answer":           result["answer"],
                "refused":          did_refuse,
                "should_refuse":    should_refuse,
                "refusal_correct":  refusal_correct,
                "chunk_1":          chunks[0] if len(chunks) > 0 else "",
                "chunk_2":          chunks[1] if len(chunks) > 1 else "",
                "chunk_3":          chunks[2] if len(chunks) > 2 else "",
                "chunk_4":          chunks[3] if len(chunks) > 3 else "",
                "context_used":     result["context_used"],
                "error":            None,
            })

            status = "⛔ REFUSED" if did_refuse else "✅ ANSWERED"
            flag = "" if refusal_correct else " ⚠️  UNEXPECTED"
            print(f"         → {status}{flag}")

        except Exception as e:
            print(f"         → ERROR: {e}")
            results.append({
                "question": question, "category": category,
                "expected_behavior": expected, "answer": "", "refused": None,
                "should_refuse": None, "refusal_correct": None,
                "chunk_1": "", "chunk_2": "", "chunk_3": "", "chunk_4": "",
                "context_used": "", "error": str(e),
            })

        time.sleep(delay_seconds)

    results_df = pd.DataFrame(results)
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅  Evaluation complete — results saved to {results_file}")
    print(f"\n📊  Refusal accuracy by category:")
    for cat in results_df["category"].unique():
        subset = results_df[results_df["category"] == cat]
        refused = subset["refused"].sum()
        total = len(subset)
        correct = subset["refusal_correct"].sum()
        print(f"    {cat:<15}: {refused}/{total} refused | {correct}/{total} correct")

    # Quick hallucination flag
    hallucinations = results_df[
        (results_df["should_refuse"] == True) & (results_df["refused"] == False)
    ]
    if len(hallucinations) > 0:
        print(f"\n⚠️  Potential hallucinations detected: {len(hallucinations)} out-of-scope")
        print("    questions answered instead of refused. Review these in RAGAS step.")

    return results_df


if __name__ == "__main__":
    run_evaluation()
