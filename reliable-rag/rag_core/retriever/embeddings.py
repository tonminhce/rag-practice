from __future__ import annotations

from typing import Any

from langchain_cohere import CohereEmbeddings

from rag_core.utils.config import EmbeddingConfig


def get_embedding_model(config: EmbeddingConfig) -> Any:
    """Instantiate an embedding model from configuration."""
    if config.provider.lower() == "cohere":
        return CohereEmbeddings(model=config.model)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")

