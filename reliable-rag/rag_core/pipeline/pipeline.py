from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

from rag_core.generator.llm import get_chat_model
from rag_core.generator.rag_chain import (
    build_hallucination_grader,
    build_highlight_chain,
    build_rag_chain,
    build_retrieval_grader,
    format_docs,
)
from rag_core.retriever.retriever import build_retriever
from rag_core.utils.config import ReliableRAGConfig, load_config
from rag_core.utils.logging import get_logger


class ReliableRAGPipeline:
    """End-to-end pipeline for running the Reliable RAG workflow."""

    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        """Initialize the pipeline from a YAML config path."""
        load_dotenv()
        self.config: ReliableRAGConfig = load_config(config_path)
        self.logger = get_logger(__name__)

        # Lazily constructed components
        self.retriever: Any | None = None
        self.generator_llm: Any | None = None
        self.grader_llm: Any | None = None
        self.rag_chain: Any | None = None
        self.retrieval_grader: Any | None = None
        self.hallucination_grader: Any | None = None
        self.highlight_chain: Any | None = None

    def setup(self) -> None:
        """Build the retriever, vector store, and LLM chains."""
        self.logger.info("Building retriever and vector store...")
        self.retriever = build_retriever(
            data_config=self.config.data,
            embedding_config=self.config.embeddings,
            retriever_config=self.config.retriever,
            persist_directory=self.config.project.persist_directory,
        )
        self.logger.info("Loading LLMs...")
        self.generator_llm = get_chat_model(self.config.llms.generator)
        self.grader_llm = get_chat_model(self.config.llms.grader)
        self.rag_chain = build_rag_chain(self.generator_llm)
        self.retrieval_grader = build_retrieval_grader(self.grader_llm)
        if self.config.evaluation.hallucination_check:
            self.hallucination_grader = build_hallucination_grader(self.grader_llm)
        if self.config.evaluation.highlight_segments:
            self.highlight_chain = build_highlight_chain(self.generator_llm)

    def run(self, question: str) -> Dict[str, Any]:
        """Run the full RAG pipeline for a single user question."""
        if self.retriever is None:
            self.setup()

        docs: List[Document] = self.retriever.invoke(question)
        filtered_docs = self._filter_docs(question, docs)
        formatted_docs = format_docs(filtered_docs)
        generation = self.rag_chain.invoke(
            {"documents": formatted_docs, "question": question}
        )

        hallucination_score: Optional[str] = None
        if self.hallucination_grader:
            hallucination_score = self.hallucination_grader.invoke(
                {"documents": formatted_docs, "generation": generation}
            ).binary_score

        highlights: Optional[Dict[str, List[str]]] = None
        if self.highlight_chain:
            lookup_response = self.highlight_chain.invoke(
                {
                    "documents": formatted_docs,
                    "question": question,
                    "generation": generation,
                }
            )
            highlights = lookup_response.dict()

        return {
            "question": question,
            "answer": generation,
            "documents_used": filtered_docs,
            "hallucination_score": hallucination_score,
            "highlights": highlights,
        }

    def _filter_docs(self, question: str, docs: List[Document]) -> List[Document]:
        """Filter retrieved documents using the relevance grader."""
        selected: List[Document] = []
        for doc in docs:
            result = self.retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            if result.binary_score.lower() == "yes":
                selected.append(doc)
        return selected or docs


def main() -> None:
    """CLI entrypoint for running the Reliable RAG pipeline."""
    parser = argparse.ArgumentParser(description="Run Reliable RAG pipeline")
    parser.add_argument("--question", required=True, help="User question to answer")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    pipeline = ReliableRAGPipeline(config_path=args.config)
    result = pipeline.run(args.question)
    logger = get_logger("cli")
    logger.info("Answer: %s", result["answer"])
    if result["hallucination_score"]:
        logger.info("Hallucination score: %s", result["hallucination_score"])


if __name__ == "__main__":
    main()

