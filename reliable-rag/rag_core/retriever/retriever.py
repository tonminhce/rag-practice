from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever

from rag_core.utils.config import DataConfig, EmbeddingConfig, RetrieverConfig
from .embeddings import get_embedding_model
from .loaders import load_from_urls
from .splitter import split_documents
from .vectorstore import build_vectorstore


def build_retriever(
    data_config: DataConfig,
    embedding_config: EmbeddingConfig,
    retriever_config: RetrieverConfig,
    persist_directory: str,
) -> VectorStoreRetriever:
    """Construct a `VectorStoreRetriever` from configuration objects.

    The function loads documents from URLs, splits them, builds an embedding-backed
    vector store, and finally exposes it as a retriever.
    """
    docs = load_from_urls(data_config.urls)
    doc_splits = split_documents(
        documents=docs,
        chunk_size=data_config.chunk_size,
        chunk_overlap=data_config.chunk_overlap,
    )
    embedding_model = get_embedding_model(embedding_config)
    vectorstore = build_vectorstore(
        documents=doc_splits,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever(
        search_type=retriever_config.type,
        search_kwargs={"k": retriever_config.k},
    )

