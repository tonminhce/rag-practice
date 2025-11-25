from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Structured output for relevance grading of retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Structured output for hallucination / grounding checks."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class HighlightDocuments(BaseModel):
    """Model capturing document segments that justify an answer."""

    id: List[str]
    title: List[str]
    source: List[str]
    segment: List[str]


def format_docs(docs: List[Document]) -> str:
    """Format documents into a tagged string for use in prompts."""
    return "\n".join(
        f"<doc{i+1}>:\nTitle:{doc.metadata.get('title', '')}\n"
        f"Source:{doc.metadata.get('source', '')}\n"
        f"Content:{doc.page_content}\n</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )


def build_retrieval_grader(llm: Any) -> Any:
    """Build a chain that grades whether a document is relevant to a question."""
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system = (
        "You are a grader assessing relevance of a retrieved document to a user "
        "question. If the document contains keyword(s) or semantic meaning related "
        "to the user question, grade it as relevant."
    )
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    return grade_prompt | structured_llm_grader


def build_rag_chain(llm: Any) -> Any:
    """Build the main RAG chain that answers questions grounded in documents."""
    system = (
        "You are an assistant for question-answering tasks.\n"
        "Use only the information in the provided documents to answer the question. "
        "If the documents do not contain enough information, say you don't know. "
        "Respond in three-to-five sentences maximum."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved documents: \n\n <docs>{documents}</docs> \n\n "
                "User question: <question>{question}</question>",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_hallucination_grader(llm: Any) -> Any:
    """Build a chain that checks if an answer is grounded in retrieved facts."""
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    system = (
        "You are a grader assessing whether an LLM generation is grounded in / "
        "supported by a set of retrieved facts. "
        "Give a binary score 'yes' or 'no'."
    )
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n <facts>{documents}</facts> \n\n "
                "LLM generation: <generation>{generation}</generation>",
            ),
        ]
    )
    return hallucination_prompt | structured_llm_grader


def build_highlight_chain(llm: Any) -> Any:
    """Build a chain that extracts verbatim segments justifying an answer."""
    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)
    system = """You are an advanced assistant for document search and retrieval.
You are provided with a question, an answer, and supporting documents. Return
verbatim segments from the documents that justify the answer."""
    prompt = PromptTemplate(
        template=(
            system
            + "\n\nUsed documents: <docs>{documents}</docs> \n\n "
            "User question: <question>{question}</question> \n\n "
            "Generated answer: <answer>{generation}</answer>\n\n"
            "{format_instructions}"
        ),
        input_variables=["documents", "question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

