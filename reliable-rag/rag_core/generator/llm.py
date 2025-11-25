from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from rag_core.utils.config import LLMConfig


def get_chat_model(config: LLMConfig) -> Any:
    """Instantiate a chat LLM client from configuration."""
    provider = config.provider.lower()
    if provider == "groq":
        return ChatGroq(model=config.model, temperature=config.temperature)
    if provider == "openai":
        return ChatOpenAI(model=config.model, temperature=config.temperature)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")

