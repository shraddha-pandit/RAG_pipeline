# RAG Pipeline Evaluation for Regulatory Document Q&A

A production-oriented Retrieval-Augmented Generation (RAG) pipeline built over SR 11-7 and BCBS 239 regulatory guidance, with a structured evaluation framework measuring faithfulness, answer relevancy, and retrieval precision.

---

## Business Context

Generative AI is being adopted rapidly across financial services, but uncontrolled language model outputs introduce a category of risk that SR 11-7 was written to address: **model risk**. When a RAG system answers questions about regulatory requirements, lending policy, or risk frameworks, a hallucinated or unsupported answer is not just inaccurate — it is a control failure.

This project applies the same rigour used in quantitative model validation to a generative AI pipeline:

- **Does the model answer only from the provided documents?** (Faithfulness)
- **Does it actually address the question asked?** (Answer Relevancy)
- **Is the retrieval step surfacing relevant material?** (Context Precision)
- **Does the system correctly refuse out-of-scope and adversarial queries?** (Refusal Accuracy)

These are not abstract research metrics. They map directly to the SR 11-7 model risk framework: conceptual soundness, ongoing monitoring, and the requirement that model outputs be explainable and auditable.

---

## Pipeline Architecture

```
PDF Documents
     │
     ▼
┌─────────────────────────────────────┐
│  Document Loading & Chunking        │
│  PyPDF · recursive text splitter    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Embedding & Vector Index           │
│  all-MiniLM-L6-v2 · ChromaDB        │
└────────────────┬────────────────────┘
                 │
          Query at runtime
                 │
                 ▼
┌─────────────────────────────────────┐
│  Retrieval                          │
│  Top-4 chunks by cosine similarity  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Augmented Generation               │
│  Groq · qwen/qwen3.8-27b            │
│  Grounded system prompt             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  RAGAS Evaluation                   │
│  Faithfulness · Answer Relevancy    │
│  Context Precision                  │
└─────────────────────────────────────┘
```

The system prompt enforces strict grounding: the model is instructed to answer **only** from retrieved context and to respond with a fixed refusal phrase when context is insufficient. This design choice makes refusal behaviour deterministic and testable.

---

## Evaluation Methodology

The evaluation set contains **17 questions** across four categories, chosen to stress-test the behaviours that matter most for enterprise deployment:

| Category | Count | Purpose |
|---|---|---|
| **Factual** | 4 | Questions with clear, verifiable answers in the documents. Baseline for retrieval and generation quality. |
| **Partial** | 4 | Questions the documents address only at a high level. Tests whether the model answers partially rather than fabricating specifics. |
| **Out-of-scope** | 5 | Questions about information not in the corpus (market data, enforcement actions, ECB guidance). The model must refuse. |
| **Adversarial** | 4 | Questions containing false premises (e.g. "SR 11-7 mandates a 12-month validation cycle — confirm this"). Tests resistance to hallucination prompting. |

This category design mirrors the testing approach in SR 11-7 Section 5 (model validation): stress testing against boundary conditions, not just measuring performance on easy cases.

---

## Key Findings

Scores were produced by RAGAS using `qwen/qwen3.8-27b` as the judge LLM and `all-MiniLM-L6-v2` for semantic similarity in answer relevancy.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| **Faithfulness** | 1.000 | ≥ 0.80 | Pass |
| **Answer Relevancy** | 0.079 | ≥ 0.70 | Fail |
| **Context Precision** | 0.361 | ≥ 0.70 | Fail |

### Faithfulness: 1.0

The model never added information that was not present in the retrieved context. Every claim in every answer was traceable to a retrieved chunk. From an MRM perspective, this is the most critical metric: a faithfulness score below 0.8 would indicate the model is generating content beyond its evidence base, which constitutes hallucination.

### Answer Relevancy: 0.079

This low score reflects an **over-refusal problem**, not a hallucination problem. The grounding system prompt, combined with the Qwen model's conservative instruction-following behaviour, caused the model to refuse factual questions it had sufficient context to answer. The refusal phrase ("I don't have enough information in the provided documents") is semantically distant from the actual question — RAGAS scores this as low relevancy. This is a calibration issue in the system prompt, not a fundamental retrieval or generation failure.

### Context Precision: 0.361

The retrieval step is returning chunks that contain relevant information, but not always in the top positions. This suggests the embedding model (`all-MiniLM-L6-v2`) is adequate for general semantic similarity but may be under-performing on domain-specific regulatory terminology. Re-ranking or a domain-adapted embedding model would likely improve this score.

### Enterprise Deployment Interpretation

For a production model risk system, the findings point to one clear remediation priority: **recalibrate the refusal threshold** in the system prompt before addressing retrieval. A model that refuses too readily fails the business use case just as surely as one that hallucinations — both produce unusable outputs. The faithfulness result confirms the generation architecture is sound; the pipeline can be trusted not to fabricate, once the over-refusal behaviour is tuned.

---

## Tech Stack

| Component | Technology |
|---|---|
| Document parsing | PyPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | ChromaDB |
| LLM (generation) | Groq API · `qwen/qwen3.8-27b` |
| LLM (RAGAS judge) | Groq API · `qwen/qwen3.8-27b` |
| Evaluation framework | RAGAS 0.2.6 |
| Orchestration | LangChain 0.3.7 |
| Language | Python 3.x |

---

## How to Run

### 1. Prerequisites

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key-here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Build the vector index

```bash
python src/load_documents.py
python src/build_vectorstore.py
```

### 4. Run the evaluation

```bash
python src/run_evaluation.py
```

Results are written to `outputs/eval_results.csv`.

### 5. Score with RAGAS

```bash
python src/evaluate_ragas.py
```

Scores are written to `outputs/ragas_scores.csv`.

---

## Folder Structure

```
rag-pipeline-review/
├── data/                        # Source PDF documents
│   ├── sr1107.pdf               # Federal Reserve SR 11-7 model risk guidance
│   ├── SR2602.pdf               # SR 26-02 model risk update
│   └── 201301-guidelines-...    # BCBS 239 risk data aggregation principles
├── src/
│   ├── load_documents.py        # PDF ingestion and chunking
│   ├── build_vectorstore.py     # Embedding and ChromaDB index creation
│   ├── rag_query.py             # RAG pipeline: retrieval + generation
│   ├── run_evaluation.py        # Runs all 17 eval questions, writes results CSV
│   ├── evaluate_ragas.py        # RAGAS scoring: faithfulness, relevancy, precision
│   └── config_comparison.py     # Utility for comparing pipeline configurations
├── outputs/
│   ├── eval_questions.csv       # 17 evaluation questions with categories
│   ├── eval_results.csv         # Raw answers and refusal flags per question
│   └── ragas_scores.csv         # RAGAS metric scores per question and category
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Documents Evaluated Against

| Document | Regulator | Scope |
|---|---|---|
| SR 11-7 (2011) | Federal Reserve | Model risk management framework for US banks |
| SR 26-02 (2026) | Federal Reserve | Updated model risk guidance |
| BCBS 239 (2013) | Basel Committee | Principles for risk data aggregation and reporting |
