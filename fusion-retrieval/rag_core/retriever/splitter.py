"""Text splitting utilities"""

from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TextCleaner(TransformComponent):
    """Transformation component to clean text by removing clutter."""

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        """Clean text in nodes by replacing tabs and paragraph separators.

        Args:
            nodes: List of nodes to clean

        Returns:
            List of cleaned nodes
        """
        for node in nodes:
            node.text = node.text.replace("\t", " ")  # Replace tabs with spaces
            node.text = node.text.replace(" \n", " ")  # Replace paragraph separator with spaces
        return nodes


class TextSplitter:
    """Handles text splitting with customizable chunk size and overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize text splitter.

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.cleaner = TextCleaner()

    def split_documents(self, documents) -> List[BaseNode]:
        """Split documents into nodes.

        Args:
            documents: List of documents to split

        Returns:
            List of split nodes
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        nodes = self.splitter.get_nodes_from_documents(documents)
        cleaned_nodes = self.cleaner(nodes)
        logger.info(f"Created {len(cleaned_nodes)} nodes after splitting")
        return cleaned_nodes


