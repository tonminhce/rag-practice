"""Document loading utilities"""

from pathlib import Path
from typing import List

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Handles loading documents from various sources."""

    @staticmethod
    def load_from_directory(
        input_dir: str, required_exts: List[str] = None
    ) -> List[Document]:
        """Load documents from a directory.

        Args:
            input_dir: Directory path containing documents
            required_exts: List of required file extensions (e.g., ['.pdf'])

        Returns:
            List of loaded documents
        """
        if required_exts is None:
            required_exts = [".pdf"]

        logger.info(f"Loading documents from {input_dir}")
        reader = SimpleDirectoryReader(input_dir=input_dir, required_exts=required_exts)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents")
        return documents

    @staticmethod
    def load_from_file(file_path: str) -> List[Document]:
        """Load a single document from file path.

        Args:
            file_path: Path to the document file

        Returns:
            List containing the loaded document
        """
        file_path_obj = Path(file_path)
        input_dir = str(file_path_obj.parent)
        required_exts = [file_path_obj.suffix]

        logger.info(f"Loading document from {file_path}")
        return DocumentLoader.load_from_directory(input_dir, required_exts)


