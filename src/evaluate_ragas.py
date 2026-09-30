"""
Step 5 — Score results using RAGAS evaluation framework.

Run from project root:
    python src/evaluate_ragas.py

Reads:  outputs/eval_results.csv
Writes: outputs/ragas_scores.csv

RAGAS Metrics explained (these are your MRM risk metrics):
─────────────────────────────────────────────────────────
  Faithfulness         — Is the answer actually supported by the retrieved chunks,
                         or did the model add information that wasn't there?
                         LOW SCORE = the model hallucinated. This is your primary risk metric.

  Answer Relevancy     — Does the answer actually address the question asked?
                         LOW SCORE = the model went off-topic or gave a vague non-answer.

  Context Precision    — Were the retrieved chunks relevant to the question?
                         LOW SCORE = the retrieval step is pulling in noise.
                         This points to embedding or chunking problems, not generation problems.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def run_ragas_evaluation(
    results_file: str = "outputs/eval_results.csv",
    scores_file: str = "outputs/ragas_scores.csv",
    api_key: str = None,
):
    """
    Score eval results using RAGAS faithfulness, answer relevancy, and context precision.
    Uses Gemini as the judge LLM (same free-tier key as the pipeline itself).
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌  GROQ_API_KEY not set in .env file.")
        sys.exit(1)

    print("📂  Loading evaluation results...")
    df = pd.read_csv(results_file)
    valid_df = df[df["error"].isna() & df["answer"].notna() & (df["answer"] != "")].copy()
    print(f"    {len(valid_df)} valid rows to score (skipping {len(df) - len(valid_df)} errors)\n")

    # Set up Groq as the RAGAS judge LLM
    # RAGAS uses the LLM to judge whether answers are grounded in context.
    print("🤖  Configuring Groq (qwen/qwen3.8-27b) as evaluation judge...")
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=api_key,
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # Build the RAGAS evaluation dataset
    print("🏗️   Building evaluation dataset...")
    samples = []
    for _, row in valid_df.iterrows():
        # Collect non-empty retrieved chunks
        retrieved = [
            row.get(f"chunk_{i}", "")
            for i in range(1, 5)
            if pd.notna(row.get(f"chunk_{i}", "")) and str(row.get(f"chunk_{i}", "")).strip()
        ]

        if not retrieved:
            # Skip rows with no retrieved context
            continue

        sample = SingleTurnSample(
            user_input=str(row["question"]),
            response=str(row["answer"]),
            retrieved_contexts=retrieved,
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    print(f"    {len(samples)} samples in evaluation dataset\n")

    # Define metrics
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]

    print("⏳  Running RAGAS evaluation (takes 3-8 minutes due to LLM judge calls)...")
    print("    Each metric requires Groq to evaluate every question-answer pair.\n")

    results = evaluate(dataset=dataset, metrics=metrics)
    scores_df = results.to_pandas()

    # Rename for clarity
    scores_df = scores_df.rename(columns={
        "llm_context_precision_without_reference": "context_precision"
    })

    # Merge scores back with the original metadata
    scored_valid = valid_df.reset_index(drop=True)
    # Only keep as many rows as we have samples (some might have been skipped)
    scored_valid = scored_valid.iloc[:len(scores_df)].copy()

    final_df = pd.concat([
        scored_valid[["question", "category", "expected_behavior", "answer", "refused", "should_refuse"]],
        scores_df[["faithfulness", "answer_relevancy", "context_precision"]],
    ], axis=1)

    Path(scores_file).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(scores_file, index=False)

    print(f"✅  Scores saved to {scores_file}")
    print(f"\n{'='*60}")
    print("📊  RAGAS SCORES SUMMARY")
    print(f"{'='*60}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    overall = final_df[metric_cols].mean()
    print(f"\nOverall averages:")
    for col in metric_cols:
        score = overall[col]
        flag = "⚠️  LOW — review carefully" if score < 0.7 else "✅"
        print(f"    {col:<30}: {score:.3f}  {flag}")

    print(f"\nScores by category:")
    cat_scores = final_df.groupby("category")[metric_cols].mean().round(3)
    print(cat_scores.to_string())

    print(f"\n🔍  Lowest faithfulness scores (potential hallucinations):")
    low_faith = final_df.nsmallest(5, "faithfulness")[
        ["question", "category", "faithfulness", "answer"]
    ]
    for _, r in low_faith.iterrows():
        print(f"    [{r['faithfulness']:.2f}] [{r['category']}] {r['question'][:60]}...")

    return final_df


if __name__ == "__main__":
    run_ragas_evaluation()
