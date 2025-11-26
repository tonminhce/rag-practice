"""Configuration management for Fusion RAG"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "text-embedding-3-small"
    dimensions: int = 512


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.1


@dataclass
class RetrieverConfig:
    """Retriever configuration."""

    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    similarity_top_k: int = 2
    num_queries: int = 1
    mode: str = "dist_based_score"


@dataclass
class DataConfig:
    """Data processing configuration."""

    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class FusionRAGConfig:
    """Top-level configuration for Fusion RAG pipeline."""

    embedding: EmbeddingConfig
    llm: LLMConfig
    retriever: RetrieverConfig
    data: DataConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> FusionRAGConfig:
        """Create config from dictionary."""
        return cls(
            embedding=EmbeddingConfig(**config_dict.get("embedding", {})),
            llm=LLMConfig(**config_dict.get("llm", {})),
            retriever=RetrieverConfig(**config_dict.get("retriever", {})),
            data=DataConfig(**config_dict.get("data", {})),
        )


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve config path relative to the Fusion RAG project root.

    We only need to walk up to the `fusion-retrieval` directory (two
    parents above this file). Using a deeper parent accidentally points
    to the repo root and breaks the lookup when running via uvicorn.
    """
    base = Path(__file__).resolve().parents[2]
    return (base / path).resolve()


def load_config(path: str = "configs/default.yaml") -> FusionRAGConfig:
    """Load configuration from YAML file."""
    config_path = _resolve_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return FusionRAGConfig.from_dict(raw)

