"""Fusion RAG pipeline combining retrieval and generation"""

from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from ..generator.llm import LLMGenerator
from ..retriever.loaders import DocumentLoader
from ..retriever.retriever import FusionRetriever
from ..retriever.splitter import TextSplitter
from ..retriever.vectorstore import VectorStoreManager
from ..utils.config import FusionRAGConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FusionRAGPipeline:
    """Main pipeline for Fusion RAG system."""

    def __init__(self, config: FusionRAGConfig):
        """Initialize Fusion RAG pipeline.

        Args:
            config: Configuration object
        """
        self.config = config

        # Set global settings
        Settings.llm = OpenAI(
            model=config.llm.model, temperature=config.llm.temperature
        )
        Settings.embed_model = OpenAIEmbedding(
            model=config.embedding.model, dimensions=config.embedding.dimensions
        )

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=config.data.chunk_size,
            chunk_overlap=config.data.chunk_overlap,
        )
        self.vector_store_manager = VectorStoreManager(config.embedding)
        self.llm_generator = LLMGenerator(config.llm)

        # Pipeline state
        self.nodes: list[BaseNode] = []
        self.vector_index = None
        self.retriever: FusionRetriever = None

        logger.info("Fusion RAG pipeline initialized")

    def ingest(self, data_path: str):
        """Ingest documents and create indexes.

        Args:
            data_path: Path to document or directory
        """
        logger.info(f"Ingesting documents from {data_path}")

        # Load documents
        documents = self.document_loader.load_from_file(data_path)

        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=[
                self.text_splitter.splitter,
                self.text_splitter.cleaner,
            ],
            vector_store=self.vector_store_manager.vector_store,
            documents=documents,
        )

        # Run pipeline to get nodes
        self.nodes = pipeline.run()
        logger.info(f"Ingestion complete: {len(self.nodes)} nodes created")

        # Create vector index
        self.vector_index = self.vector_store_manager.create_index(self.nodes)
        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=self.config.retriever.similarity_top_k
        )

        # Create fusion retriever
        self.retriever = FusionRetriever(
            nodes=self.nodes,
            vector_retriever=vector_retriever,
            retriever_config=self.config.retriever,
        )

        logger.info("Pipeline ready for queries")

    def query(self, query: str) -> dict:
        """Query the RAG pipeline.

        Args:
            query: Query string

        Returns:
            Dictionary containing answer, context, and query
        """
        if self.retriever is None:
            raise ValueError("Pipeline not initialized. Call ingest() first.")

        logger.info(f"Processing query: {query}")

        # Retrieve relevant nodes
        retrieved_nodes = self.retriever.retrieve(query)

        # Extract context from retrieved nodes
        context = "\n\n".join([node.text for node in retrieved_nodes])

        # Generate answer using LLM
        prompt = f"""Based on the following context, please answer the question.
        
Context:
{context}

Question: {query}

Answer:"""

        answer = self.llm_generator.generate(prompt)

        result = {
            "query": query,
            "answer": answer,
            "context": [node.text for node in retrieved_nodes],
            "scores": [node.score for node in retrieved_nodes] if hasattr(retrieved_nodes[0], 'score') else None,
        }

        logger.info("Query processed successfully")
        return result

