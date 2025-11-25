from __future__ import annotations

from pathlib import Path
from typing import Any, List

from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_vectorstore(
    documents: List[Document],
    embedding: Any,
    collection_name: str = "rag",
    persist_directory: str | None = None,
) -> Chroma:
    """Create a Chroma vector store from documents and an embedding model.

    Args:
        documents: Split documents to index.
        embedding: LangChain-compatible embedding model instance.
        collection_name: Name of the Chroma collection.
        persist_directory: Optional directory for on-disk persistence.

    Returns:
        A Chroma vector store instance.
    """
    persist_path = (
        Path(persist_directory).as_posix() if persist_directory is not None else None
    )
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_path,
    )

