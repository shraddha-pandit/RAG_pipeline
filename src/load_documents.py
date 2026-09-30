"""
Step 1 — Load PDFs from data/ and split them into chunks.

Run from project root:
    python src/load_documents.py
"""

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_documents(
    data_dir: str = "data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True,
) -> list:
    """
    Load all PDFs from data_dir and split into overlapping chunks.

    Args:
        data_dir:      Folder containing your PDF files
        chunk_size:    Max characters per chunk (smaller = more precise retrieval,
                       larger = more context per chunk — a key tradeoff to evaluate)
        chunk_overlap: Characters shared between adjacent chunks so meaning isn't
                       cut off at boundaries
        verbose:       Print progress

    Returns:
        List of LangChain Document objects (each = one chunk)
    """
    data_path = Path(data_dir)
    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        print(f"❌  No PDFs found in '{data_dir}/'")
        print("    Download the three recommended documents and put them in data/")
        print("    See README.md for direct download links.")
        sys.exit(1)

    if verbose:
        print(f"📄  Found {len(pdf_files)} PDF(s):")

    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        all_docs.extend(docs)
        if verbose:
            print(f"    • {pdf_file.name}  ({len(docs)} pages)")

    # RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    # then sentences, then words — preserving natural language structure.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    if verbose:
        print(f"\n✅  Chunking complete")
        print(f"    Total pages  : {len(all_docs)}")
        print(f"    Total chunks : {len(chunks)}")
        print(f"    Chunk size   : {chunk_size} chars, Overlap: {chunk_overlap} chars")
        print(f"\n📝  Sample chunk (first 500 chars):")
        print("─" * 60)
        print(chunks[0].page_content[:500])
        print(f"─" * 60)
        print(f"    Metadata: {chunks[0].metadata}")

    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"\nDone. {len(chunks)} chunks ready for embedding.")
