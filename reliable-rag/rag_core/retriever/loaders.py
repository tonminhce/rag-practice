from __future__ import annotations

from typing import Iterable, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_from_urls(urls: Iterable[str]) -> List[Document]:
    """Load documents from a collection of URLs using `WebBaseLoader`."""
    documents: List[Document] = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    return documents

