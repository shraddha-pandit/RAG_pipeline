"""
Step 6 — Compare retrieval quality across chunking and embedding configurations.

This is the 'compared retrieval quality across chunking and embedding approaches'
bullet on your CV — actually run it so you can speak to the results.

Run from project root:
    python src/config_comparison.py

Writes: outputs/config_comparison.csv

Configurations tested:
    A — chunk_size=1000, model=all-MiniLM-L6-v2  (baseline)
    B — chunk_size=500,  model=all-MiniLM-L6-v2  (smaller chunks)
    C — chunk_size=1000, model=all-mpnet-base-v2  (stronger embedding model)

What to look for in the results:
    • Config B (smaller chunks) — typically better context_precision (more focused
      retrieval) but potentially lower context_recall (may miss broader context)
    • Config C (better embeddings) — typically better semantic matching, especially
      for paraphrased or indirect questions
    • Any config that dramatically changes faithfulness is worth noting — it means
      retrieval quality is affecting whether the LLM hallucinates
"""

import os
import sys
import shutil
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from build_vectorstore import build_vectorstore
from rag_query import get_rag_answer

CONFIGS = [
    {
        "name": "A_baseline",
        "label": "Baseline (1000 chars, MiniLM)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "all-MiniLM-L6-v2",
        "vectorstore_dir": "vectorstore_A",
    },
    {
        "name": "B_small_chunks",
        "label": "Small chunks (500 chars, MiniLM)",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_model": "all-MiniLM-L6-v2",
        "vectorstore_dir": "vectorstore_B",
    },
    {
        "name": "C_mpnet",
        "label": "Larger model (1000 chars, MPNet)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "all-mpnet-base-v2",
        "vectorstore_dir": "vectorstore_C",
    },
]

# Use a representative subset of questions for comparison (saves API costs)
COMPARISON_QUESTIONS = [
    "What is model risk and how do financial institutions manage it?",
    "What are the three types of model error described in the guidance?",
    "What is the current inflation rate in the United States?",  # Should refuse
    "What specific dollar threshold triggers mandatory model validation review?",  # Adversarial
    "What does the guidance say about the role of senior management in model risk?",
    "How often should models be validated according to the documents?",
    "What machine learning techniques are recommended for credit risk models?",
    "What data governance principles are outlined for risk data aggregation?",
]


def build_all_configs(data_dir: str = "data", force_rebuild: bool = False):
    """Build vectorstores for all three configurations."""
    for cfg in CONFIGS:
        vdir = cfg["vectorstore_dir"]
        if Path(vdir).exists() and not force_rebuild:
            print(f"⏭️   Config {cfg['name']}: vectorstore exists, skipping rebuild")
            continue

        if Path(vdir).exists():
            shutil.rmtree(vdir)

        print(f"\n🔨  Building config {cfg['name']}: {cfg['label']}")
        build_vectorstore(
            data_dir=data_dir,
            persist_dir=vdir,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            embedding_model=cfg["embedding_model"],
            verbose=True,
        )


def run_comparison(
    output_file: str = "outputs/config_comparison.csv",
    api_key: str = None,
    delay_seconds: float = 1.5,
):
    """Run the comparison questions through all three configs and save results."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌  GEMINI_API_KEY not set.")
        sys.exit(1)

    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'─'*60}")
        print(f"📐  Running config: {cfg['label']}")
        print(f"{'─'*60}")

        for q in COMPARISON_QUESTIONS:
            print(f"  Q: {q[:65]}...")
            try:
                result = get_rag_answer(
                    q,
                    vectorstore_dir=cfg["vectorstore_dir"],
                    embedding_model=cfg["embedding_model"],
                    api_key=api_key,
                )
                all_results.append({
                    "config_name":     cfg["name"],
                    "config_label":    cfg["label"],
                    "chunk_size":      cfg["chunk_size"],
                    "embedding_model": cfg["embedding_model"],
                    "question":        q,
                    "answer":          result["answer"],
                    "refused":         result["refused"],
                    "chunk_1":         result["retrieved_chunks"][0] if result["retrieved_chunks"] else "",
                    "chunk_2":         result["retrieved_chunks"][1] if len(result["retrieved_chunks"]) > 1 else "",
                    "error":           None,
                })
                status = "⛔ refused" if result["refused"] else "✅ answered"
                print(f"     → {status}")

            except Exception as e:
                print(f"     → ERROR: {e}")
                all_results.append({
                    "config_name": cfg["name"], "config_label": cfg["label"],
                    "chunk_size": cfg["chunk_size"], "embedding_model": cfg["embedding_model"],
                    "question": q, "answer": "", "refused": None,
                    "chunk_1": "", "chunk_2": "", "error": str(e),
                })

            time.sleep(delay_seconds)

    df = pd.DataFrame(all_results)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅  Comparison saved to {output_file}")
    print(f"\n📊  Refusal rate by config (out-of-scope question: 'current inflation rate'):")
    out_of_scope = df[df["question"].str.contains("inflation")]
    for cfg_name in df["config_name"].unique():
        subset = out_of_scope[out_of_scope["config_name"] == cfg_name]
        refused = subset["refused"].any()
        print(f"    {cfg_name}: {'✅ Refused (correct)' if refused else '⚠️  Answered (incorrect)'}")

    print(f"\n📊  Refusal rate by config (all questions):")
    summary = df.groupby(["config_name", "config_label"])["refused"].agg(["sum", "count"])
    summary["refusal_rate"] = (summary["sum"] / summary["count"]).round(3)
    print(summary[["refusal_rate"]].to_string())

    return df


def score_comparison_with_ragas(
    comparison_file: str = "outputs/config_comparison.csv",
    scored_file: str = "outputs/config_comparison_scored.csv",
    api_key: str = None,
):
    """
    Run RAGAS faithfulness and context_precision on comparison results.
    This gives quantitative metrics for the config comparison table.
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    df = pd.read_csv(comparison_file)

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    )

    scored_rows = []
    for cfg_name in df["config_name"].unique():
        cfg_df = df[(df["config_name"] == cfg_name) & df["error"].isna() & (df["answer"] != "")].copy()
        print(f"\n⏳  Scoring config {cfg_name} ({len(cfg_df)} questions)...")

        samples = []
        for _, row in cfg_df.iterrows():
            retrieved = [c for c in [row.get("chunk_1", ""), row.get("chunk_2", "")] if c]
            if not retrieved:
                continue
            samples.append(SingleTurnSample(
                user_input=str(row["question"]),
                response=str(row["answer"]),
                retrieved_contexts=retrieved,
            ))

        if not samples:
            continue

        dataset = EvaluationDataset(samples=samples)
        results = evaluate(dataset=dataset, metrics=[
            Faithfulness(llm=evaluator_llm),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        ])
        scores = results.to_pandas()

        cfg_df = cfg_df.iloc[:len(scores)].copy().reset_index(drop=True)
        cfg_df["faithfulness"] = scores["faithfulness"].values
        cfg_df["context_precision"] = scores.get(
            "llm_context_precision_without_reference", pd.Series([None]*len(scores))
        ).values
        scored_rows.append(cfg_df)

    if scored_rows:
        final_df = pd.concat(scored_rows, ignore_index=True)
        final_df.to_csv(scored_file, index=False)

        print(f"\n✅  Scored comparison saved to {scored_file}")
        print(f"\n📊  Config comparison table:")
        summary = final_df.groupby(["config_name", "config_label"])[
            ["faithfulness", "context_precision"]
        ].mean().round(3)
        print(summary.to_string())
        return final_df

    print("No scored rows produced. Check for errors above.")
    return None


if __name__ == "__main__":
    print("🔧  Phase 1: Building all three vectorstore configurations...")
    build_all_configs()

    print("\n🔍  Phase 2: Running comparison questions through all configs...")
    run_comparison()

    print("\n📈  Phase 3: Scoring with RAGAS (optional — takes ~5 mins)...")
    score = input("Run RAGAS scoring now? (y/n): ").strip().lower()
    if score == "y":
        score_comparison_with_ragas()
    else:
        print("Skipped. Run score_comparison_with_ragas() manually when ready.")
