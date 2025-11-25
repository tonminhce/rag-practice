from langchain_core.documents import Document

from rag_core.generator.rag_chain import format_docs


def test_format_docs_contains_metadata():
    docs = [
        Document(
            page_content="content",
            metadata={"title": "Title", "source": "http://example.com"},
        )
    ]
    formatted = format_docs(docs)
    assert "Title:Title" in formatted
    assert "content" in formatted

