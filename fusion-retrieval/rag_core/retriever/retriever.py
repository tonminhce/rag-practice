"""Fusion retriever combining vector and BM25 retrieval"""

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever

from ..utils.config import RetrieverConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FusionRetriever:
    """Fusion retriever combining vector-based and BM25 keyword-based retrieval."""

    def __init__(
        self,
        nodes: list[BaseNode],
        vector_retriever,
        retriever_config: RetrieverConfig,
    ):
        """Initialize fusion retriever.

        Args:
            nodes: List of nodes for BM25 indexing
            vector_retriever: Vector-based retriever instance
            retriever_config: Retriever configuration
        """
        self.config = retriever_config
        self.nodes = nodes

        # Create BM25 retriever
        logger.info("Creating BM25 retriever")
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=retriever_config.similarity_top_k
        )

        # Create fusion retriever
        logger.info("Creating fusion retriever")
        self.retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            retriever_weights=[
                retriever_config.vector_weight,
                retriever_config.bm25_weight,
            ],
            num_queries=retriever_config.num_queries,
            mode=retriever_config.mode,
            use_async=False,
        )
        logger.info("Fusion retriever created successfully")

    def retrieve(self, query: str):
        """Retrieve relevant nodes for a query.

        Args:
            query: Query string

        Returns:
            List of retrieved nodes with scores
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        results = self.retriever.retrieve(query)
        logger.info(f"Retrieved {len(results)} documents")
        return results


