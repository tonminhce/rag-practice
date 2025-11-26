# Fusion Retrieval RAG

A clean, production-ready implementation of Fusion Retrieval-Augmented Generation (RAG) system that combines vector-based similarity search with keyword-based BM25 retrieval using LlamaIndex.

## Overview

This project implements a Fusion Retrieval system that leverages both semantic understanding (vector-based) and keyword matching (BM25) to improve document retrieval quality. The system is designed with a clean, modular architecture suitable for production use.

## Features

- **Hybrid Retrieval**: Combines vector-based (FAISS) and keyword-based (BM25) retrieval
- **LlamaIndex Integration**: Built on LlamaIndex for robust document processing
- **RESTful API**: FastAPI-based API for easy integration
- **Configurable**: YAML-based configuration for easy customization
- **Clean Architecture**: Modular design with separation of concerns

## Project Structure

```
fusion-retrieval/
├── api/                    # FastAPI application
│   └── main.py            # API endpoints
├── configs/               # Configuration files
│   └── default.yaml       # Default configuration
├── data/                  # Input data storage
│   └── Understanding_Climate_Change.pdf
├── eval/                  # Evaluation scripts
├── notebooks/             # Jupyter notebooks for experimentation
│   └── fusion_retrieval_with_llamaindex.ipynb
├── rag_core/              # Core RAG implementation
│   ├── generator/         # LLM generation
│   │   └── llm.py
│   ├── pipeline/          # Main pipeline
│   │   └── pipeline.py
│   ├── retriever/         # Retrieval components
│   │   ├── loaders.py
│   │   ├── retriever.py
│   │   ├── splitter.py
│   │   └── vectorstore.py
│   └── utils/             # Utilities
│       ├── config.py
│       └── logging.py
├── .env                   # Environment variables (create this)
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository and navigate to the fusion-retrieval directory:
```bash
cd fusion-retrieval
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Embedding**: Model and dimensions
- **LLM**: Model and temperature
- **Retriever**: Weights, top-k, fusion mode
- **Data**: Chunk size and overlap

### Retriever Modes

- `dist_based_score`: MinMax scaling based on mean and std (recommended)
- `relative_score`: MinMax based on min/max scores
- `reciprocal_rerank`: Reciprocal ranking
- `simple`: Maximum score method

## Usage

### API Server

Start the FastAPI server:

```bash
python -m api.main
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Ingest Documents
```bash
POST /ingest
Content-Type: application/json

{
  "data_path": "data/Understanding_Climate_Change.pdf"
}
```

#### 3. Query
```bash
POST /query
Content-Type: application/json

{
  "query": "What are the impacts of climate change on the environment?"
}
```

### Programmatic Usage

```python
from rag_core.pipeline.pipeline import FusionRAGPipeline
from rag_core.utils.config import load_config

# Load configuration
config = load_config()

# Initialize pipeline
pipeline = FusionRAGPipeline(config)

# Ingest documents
pipeline.ingest("data/Understanding_Climate_Change.pdf")

# Query
result = pipeline.query("What are the impacts of climate change?")
print(result["answer"])
```

## Development

### Running Notebooks

Jupyter notebooks in the `notebooks/` directory can be used for experimentation and prototyping.

### Evaluation

Evaluation scripts can be added to the `eval/` directory for assessing RAG performance.

