"""
Step 2 — Embed document chunks and store in ChromaDB.

Run from project root:
    python src/build_vectorstore.py

This creates a persistent vector database in vectorstore/
You only need to run this once (or again if you change config).
"""

import sys
from pathlib import Path

# Allow importing from src/ when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from load_documents import load_and_chunk_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore(
    data_dir: str = "data",
    persist_dir: str = "vectorstore",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2",
    verbose: bool = True,
):
    """
    Create and persist a ChromaDB vector store from documents in data_dir.

    Embedding model choices (used in config_comparison.py):
        "all-MiniLM-L6-v2"  — fast, compact, good general performance (384 dims)
        "all-mpnet-base-v2" — slower, larger, often better quality (768 dims)

    Args:
        data_dir:        Folder with source PDFs
        persist_dir:     Where to save the ChromaDB files
        chunk_size:      Characters per chunk (passed to loader)
        chunk_overlap:   Overlap between chunks (passed to loader)
        embedding_model: Sentence-transformers model name
        verbose:         Print progress

    Returns:
        Chroma vectorstore instance
    """
    if verbose:
        print(f"🔧  Building vectorstore")
        print(f"    Embedding model : {embedding_model}")
        print(f"    Chunk size      : {chunk_size} | Overlap: {chunk_overlap}")
        print(f"    Output dir      : {persist_dir}/\n")

    chunks = load_and_chunk_documents(data_dir, chunk_size, chunk_overlap, verbose)

    if verbose:
        print(f"\n🔢  Creating embeddings (downloading model on first run — ~90MB)...")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build and persist the vector database
    # Each chunk's text is turned into a dense vector; ChromaDB indexes them
    # for fast approximate nearest-neighbour search at query time.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    count = vectorstore._collection.count()

    if verbose:
        print(f"\n✅  Vectorstore built")
        print(f"    Vectors stored  : {count}")
        print(f"    Persisted to    : {persist_dir}/")

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
    print("\nDone. Vectorstore ready — run rag_query.py to test retrieval.")
