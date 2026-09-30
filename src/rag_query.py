"""
Step 3 — Query the RAG pipeline.

Run from project root to test 3 sample questions:
    python src/rag_query.py

Requires: GROQ_API_KEY environment variable
    export GROQ_API_KEY="your-key-here"
"""

import os
import sys
from pathlib import Path

from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── System prompt ─────────────────────────────────────────────────────────────
# This is the critical grounding instruction. Whether the model ACTUALLY obeys it
# is what your faithfulness evaluation tests. Weak adherence here = hallucination.
SYSTEM_PROMPT = """You are a precise document analyst. Your role is to answer questions
based ONLY on the context documents provided below.

RULES — you must follow these without exception:
1. Use only information explicitly present in the provided context.
2. Do not use any prior knowledge, training data, or external information.
3. If the context does not contain sufficient information to answer the question,
   respond with exactly:
   "I don't have enough information in the provided documents to answer this question."
4. Do not infer, extrapolate, or speculate beyond what the context states.
5. If you reference specific information, indicate which part of the context it came from.
6. Never fabricate facts, statistics, dates, or named sources."""


def get_rag_answer(
    question: str,
    vectorstore_dir: str = "vectorstore",
    embedding_model: str = "all-MiniLM-L6-v2",
    top_k: int = 4,
    api_key: str = None,
) -> dict:
    """
    Run a question through the full RAG pipeline and return the answer + context.

    Pipeline steps:
        1. Embed the question with the same model used for documents
        2. Retrieve the top_k most semantically similar chunks from ChromaDB
        3. Inject those chunks into the prompt sent to Gemini
        4. Return the answer + the retrieved chunks (for grounding evaluation)

    Args:
        question:        The user's question
        vectorstore_dir: Path to the persisted ChromaDB directory
        embedding_model: Must match the model used in build_vectorstore.py
        top_k:           Number of chunks to retrieve (more = more context, more noise)
        api_key:         Groq API key (falls back to GROQ_API_KEY env var)

    Returns:
        dict with keys: question, answer, retrieved_chunks, retrieved_sources,
                        context_used, refused
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key not found.\n"
            "Set GROQ_API_KEY in your .env file."
        )

    # Load vectorstore (must match the embedding model used at build time)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings,
    )

    # Retrieve top_k most relevant chunks using cosine similarity
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    # Build the context string to inject into the prompt
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build the full prompt
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT DOCUMENTS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    # Call Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    # Flag whether the model declined to answer
    refused = "i don't have enough information" in answer.lower()

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": [doc.page_content for doc in retrieved_docs],
        "retrieved_sources": [doc.metadata for doc in retrieved_docs],
        "context_used": context,
        "refused": refused,
    }


if __name__ == "__main__":
    sample_questions = [
        "What is model risk and why does it matter to financial institutions?",
        "What are the key components of an effective model validation framework?",
        "What is the current federal funds rate?",  # Should trigger refusal
    ]

    print("🔍  Testing RAG pipeline with 3 sample questions\n")
    print("=" * 70)

    for q in sample_questions:
        print(f"\nQ: {q}")
        result = get_rag_answer(q)
        print(f"A: {result['answer'][:400]}")
        print(f"   {'⛔ Model refused (correct for out-of-scope)' if result['refused'] else '📋 Model answered from context'}")
        print(f"   Retrieved {len(result['retrieved_chunks'])} chunks")
        print("-" * 70)
