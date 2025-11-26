"""LLM generation utilities"""

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from ..utils.config import LLMConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """Handles LLM-based text generation."""

    def __init__(self, llm_config: LLMConfig):
        """Initialize LLM generator.

        Args:
            llm_config: LLM configuration
        """
        self.config = llm_config
        self.llm = OpenAI(model=llm_config.model, temperature=llm_config.temperature)
        Settings.llm = self.llm
        logger.info(f"Initialized LLM: {llm_config.model}")

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        logger.info("Generating response from LLM")
        response = self.llm.complete(prompt)
        return str(response)

    def get_llm(self):
        """Get the underlying LLM instance.

        Returns:
            LLM instance
        """
        return self.llm

