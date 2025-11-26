"""Vector store management"""

import faiss
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from ..utils.config import EmbeddingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages vector store creation and indexing."""

    def __init__(self, embed_config: EmbeddingConfig):
        """Initialize vector store manager.

        Args:
            embed_config: Embedding configuration
        """
        self.embed_config = embed_config
        self.embed_model = OpenAIEmbedding(
            model=embed_config.model, dimensions=embed_config.dimensions
        )
        self.faiss_index = faiss.IndexFlatL2(embed_config.dimensions)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)

    def create_index(self, nodes: list[BaseNode]) -> VectorStoreIndex:
        """Create a vector store index from nodes.

        Args:
            nodes: List of nodes to index

        Returns:
            VectorStoreIndex instance
        """
        logger.info(f"Creating vector store index from {len(nodes)} nodes")
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        logger.info("Vector store index created successfully")
        return index

    def get_vector_store(self) -> FaissVectorStore:
        """Get the underlying FAISS vector store.

        Returns:
            FaissVectorStore instance
        """
        return self.vector_store


