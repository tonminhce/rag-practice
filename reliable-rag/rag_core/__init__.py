"""
Reliable RAG core package.

The submodules are split across retriever, generator, utils, and pipeline to
mirror the sections of the original notebook.
"""

from .pipeline.pipeline import ReliableRAGPipeline

__all__ = ["ReliableRAGPipeline"]

