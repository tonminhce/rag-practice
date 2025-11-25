from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ProjectConfig:
    """Project-level settings such as name and persistence directory."""

    name: str
    persist_directory: str


@dataclass
class DataConfig:
    """Configuration for data ingestion and chunking."""

    urls: List[str]
    chunk_size: int
    chunk_overlap: int


@dataclass
class EmbeddingConfig:
    """Embedding provider and model configuration."""

    provider: str
    model: str


@dataclass
class RetrieverConfig:
    """Retriever strategy and top-k configuration."""

    type: str
    k: int


@dataclass
class LLMConfig:
    """Single LLM configuration (provider, model, temperature)."""

    provider: str
    model: str
    temperature: float


@dataclass
class LLMSettings:
    """Grouped LLM settings for generator and grader models."""

    generator: LLMConfig
    grader: LLMConfig


@dataclass
class EvaluationConfig:
    """Toggles for evaluation behaviors (hallucination, highlighting)."""

    hallucination_check: bool
    highlight_segments: bool


@dataclass
class ReliableRAGConfig:
    """Top-level configuration object for the Reliable RAG pipeline."""

    project: ProjectConfig
    data: DataConfig
    embeddings: EmbeddingConfig
    retriever: RetrieverConfig
    llms: LLMSettings
    evaluation: EvaluationConfig


def _resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to the project root (two levels above this file)."""
    base = Path(__file__).resolve().parents[2]
    return (base / path).resolve()


def load_config(path: str = "configs/default.yaml") -> ReliableRAGConfig:
    """Load a YAML config file into a `ReliableRAGConfig` instance."""
    config_path = _resolve(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    evaluation_raw = raw["evaluation"]
    # Support both the corrected key and the original misspelled one.
    hallucination_check = evaluation_raw.get(
        "hallucination_check", evaluation_raw.get("hallucinaton_check", False)
    )

    return ReliableRAGConfig(
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(**raw["data"]),
        embeddings=EmbeddingConfig(**raw["embeddings"]),
        retriever=RetrieverConfig(**raw["retriever"]),
        llms=LLMSettings(
            generator=LLMConfig(**raw["llms"]["generator"]),
            grader=LLMConfig(**raw["llms"]["grader"]),
        ),
        evaluation=EvaluationConfig(
            hallucination_check=hallucination_check,
            highlight_segments=evaluation_raw["highlight_segments"],
        ),
    )

