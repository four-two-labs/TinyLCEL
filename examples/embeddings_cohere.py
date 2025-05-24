"""Example of using CohereEmbeddings."""

import os
import asyncio

from dotenv import load_dotenv

from tinylcel.embeddings.cohere import CohereEmbeddings


async def main():
    # Load environment variables from .env file (optional)
    # Ensure COHERE_API_KEY is set in your environment or .env file
    load_dotenv()

    # Check if the API key is available
    if not os.environ.get('COHERE_API_KEY'):
        print("COHERE_API_KEY not found in environment variables.")
        print("Please set it or create a .env file with COHERE_API_KEY=your-key")
        return

    # --- Basic Usage --- #
    print("--- Basic Usage ---")
    embedder = CohereEmbeddings(
        model='embed-v4.0',
        api_key=os.environ.get('COHERE_API_KEY'),
        base_url=os.environ.get('COHERE_BASE_URL'),
        dimensions=256
    )    

    query = "What is the capital of France?"
    query_embedding = embedder.embed_query(query)
    print(f"Query: \"{query}\"")
    print(f"Embedding dimension: {len(query_embedding)}")
    print(f"First 5 elements: {query_embedding[:5]}")
    print("---")

    doc_texts = [
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is an iron lattice tower located on the Champ de Mars in Paris.",
        "French cuisine is known for its cheeses and wines."
    ]
    doc_embeddings = embedder.embed_documents(doc_texts)
    print(f"Documents to embed: {len(doc_texts)}")
    print(f"Number of embeddings generated: {len(doc_embeddings)}")
    print(f"Dimension of first doc embedding: {len(doc_embeddings[0])}")
    print(f"First 5 elements of first doc embedding: {doc_embeddings[0][:5]}")
    print("---")

    # --- Async Usage --- #
    print("\n--- Async Usage ---")
    async_query_embedding = await embedder.aembed_query(query)
    print(f"Async Query: \"{query}\"")
    print(f"Async Embedding dimension: {len(async_query_embedding)}")
    print(f"Async First 5 elements: {async_query_embedding[:5]}")
    # Verify async and sync produce the same result (should be deterministic)
    assert query_embedding == async_query_embedding
    print("Async query embedding matches sync embedding.")
    print("---")

    async_doc_embeddings = await embedder.aembed_documents(doc_texts)
    print(f"Async Documents to embed: {len(doc_texts)}")
    print(f"Async Number of embeddings generated: {len(async_doc_embeddings)}")
    print(f"Async Dimension of first doc embedding: {len(async_doc_embeddings[0])}")
    print(f"Async First 5 elements of first doc embedding: {async_doc_embeddings[0][:5]}")
    # Verify async and sync produce the same result
    assert doc_embeddings == async_doc_embeddings
    print("Async document embeddings match sync embeddings.")
    print("---")


if __name__ == "__main__":
    # Setup instructions:
    # 1. Install dependencies: pip install -e .[cohere] python-dotenv
    # 2. Create a .env file in the root directory with: COHERE_API_KEY=your_cohere_api_key
    # 3. Run this script from the root directory: python examples/embeddings_cohere.py
    asyncio.run(main()) 