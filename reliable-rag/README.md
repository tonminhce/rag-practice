# Reliable RAG Service

Modular production-ready layout derived from the exploratory `reliable_rag.ipynb`.
The codebase exposes reusable building blocks for loading documents, building a
vector store, scoring retrieved chunks, generating answers, and evaluating
outputs.

## Directory Layout

- `README.md`, `requirements.txt`, `.gitignore`: project metadata and dependency lock.
- `rag_core/`: Python package that implements the RAG pipeline.
  - `retriever/`: ingestion, splitting, vector store, retriever helpers.
  - `generator/`: LLM wrappers, prompts, QA + evaluation chains.
  - `pipeline/`: orchestration to run the full workflow end-to-end.
  - `utils/`: configuration loading, logging, shared helpers.
- `configs/`: YAML config for URLs, embedding models, retriever parameters, LLMs.
- `documents/`: persisted raw documents (local copies, PDFs, etc.).
- `data/`: intermediate artifacts (vector-store persistence, cache).
- `api/`: entry points for serving the pipeline (FastAPI/Streamlit placeholder).
- `eval/`: scripts/notebooks for scoring relevance/faithfulness.
- `notebooks/`: experimentation notebooks (`reliable_rag.ipynb` lives here).
- `tests/`: unit tests for the core package.

## Getting Started

```bash
python -m venv .venv && source .venv/Scripts/activate  # PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # set GROQ_API_KEY, OPENAI_API_KEY, COHERE_API_KEY
python -m rag_core.pipeline.pipeline --question "What are agentic design patterns?"
```

## Environment Variables

The pipeline relies on the same keys that the original notebook required:

| Variable          | Purpose                               |
|-------------------|---------------------------------------|
| `GROQ_API_KEY`    | ChatGroq client (llama-3.1-8b-instant) |
| `OPENAI_API_KEY`  | OpenAI client (gpt-4o-mini graders)    |
| `COHERE_API_KEY`  | Cohere embeddings (embed-english-v3.0) |

## Notes

- The `configs/default.yaml` file mirrors the notebook defaults. Update URLs or
  model names there instead of editing code.
- Tests illustrate how to mock external services so you can add coverage without
  hitting real APIs.
- The pipeline keeps the separation between retrieval, grading, answer
  generation, hallucination checks, and highlighting, making it easy to swap any
  component.

