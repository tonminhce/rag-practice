from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into smaller overlapping chunks using a token-aware splitter.

    Args:
        documents: Input LangChain documents to split.
        chunk_size: Target size of each chunk (in tokens, approximately).
        chunk_overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        A list of split `Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)

