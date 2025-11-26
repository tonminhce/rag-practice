"""Example usage of Fusion RAG Pipeline"""

import os
from dotenv import load_dotenv
from rag_core.pipeline.pipeline import FusionRAGPipeline
from rag_core.utils.config import load_config

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")


def main():
    """Example usage of the Fusion RAG pipeline."""
    # Load configuration
    config = load_config()

    # Initialize pipeline
    print("Initializing Fusion RAG pipeline...")
    pipeline = FusionRAGPipeline(config)

    # Ingest documents
    data_path = "data/Understanding_Climate_Change.pdf"
    print(f"\nIngesting documents from {data_path}...")
    pipeline.ingest(data_path)

    # Query the pipeline
    query = "What are the impacts of climate change on the environment?"
    print(f"\nQuerying: {query}")
    result = pipeline.query(query)

    # Display results
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXT:")
    print("=" * 80)
    for i, context in enumerate(result["context"], 1):
        print(f"\nContext {i}:")
        print(context[:200] + "..." if len(context) > 200 else context)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


