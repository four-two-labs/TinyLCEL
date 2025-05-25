"""Example of using OpenAIEmbeddings."""

import os
import asyncio

from dotenv import load_dotenv

from tinylcel.providers.openai import OpenAIEmbeddings


async def main():
    load_dotenv()

    if not os.environ.get('OPENAI_API_KEY'):
        print("OPENAI_API_KEY not found. Please set it or create a .env file.")
        return

    print("--- Basic Usage (text-embedding-ada-002) ---")
    ada_embedder = OpenAIEmbeddings() # Defaults to text-embedding-ada-002

    query = "What is the best search engine?"
    query_embedding_ada = ada_embedder.embed_query(query)
    print(f"Query (ada): \"{query}\"")
    print(f"Embedding dimension (ada): {len(query_embedding_ada)}")
    print(f"First 5 elements (ada): {query_embedding_ada[:5]}")
    print("---")

    doc_texts = [
        "Google is the most popular search engine worldwide.",
        "Bing is Microsoft's answer to Google Search.",
        "DuckDuckGo is a privacy-focused search engine."
    ]
    doc_embeddings_ada = ada_embedder.embed_documents(doc_texts)
    print(f"Documents to embed (ada): {len(doc_texts)}")
    print(f"Number of embeddings generated (ada): {len(doc_embeddings_ada)}")
    print(f"Dimension of first doc embedding (ada): {len(doc_embeddings_ada[0])}")
    print(f"First 5 elements of first doc embedding (ada): {doc_embeddings_ada[0][:5]}")
    print("---")

    print("\n--- Async Usage (text-embedding-ada-002) ---")
    async_query_embedding_ada = await ada_embedder.aembed_query(query)
    assert query_embedding_ada == async_query_embedding_ada
    print("Async query embedding (ada) matches sync.")

    async_doc_embeddings_ada = await ada_embedder.aembed_documents(doc_texts)
    assert doc_embeddings_ada == async_doc_embeddings_ada
    print("Async document embeddings (ada) match sync.")
    print("---")

    # --- Usage with a newer model supporting `dimensions` ---
    # Note: text-embedding-3-small can output 512 or 1536 dimensions (default)
    # text-embedding-3-large can output 256, 1024 or 3072 dimensions (default)
    print("\n--- Advanced Usage (text-embedding-3-large with custom dimensions) ---")
    try:
        large_embedder = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=512)
        
        query_embedding_large = large_embedder.embed_query(query)
        print(f"Query (3-large, 512d): \"{query}\"")
        print(f"Embedding dimension (3-large, 512d): {len(query_embedding_large)}")
        print(f"First 5 elements (3-large, 512d): {query_embedding_large[:5]}")
        print("---")

        doc_embeddings_large = large_embedder.embed_documents(doc_texts)
        print(f"Documents to embed (3-large, 512d): {len(doc_texts)}")
        print(f"Number of embeddings generated (3-large, 512d): {len(doc_embeddings_large)}")
        print(f"Dimension of first doc embedding (3-large, 512d): {len(doc_embeddings_large[0])}")
        print("---")

        print("\n--- Async Usage (text-embedding-3-large with custom dimensions) ---")
        async_query_embedding_large = await large_embedder.aembed_query(query)
        assert query_embedding_large == async_query_embedding_large
        print("Async query embedding (3-large, 512d) matches sync.")
        async_doc_embeddings_large = await large_embedder.aembed_documents(doc_texts)
        assert doc_embeddings_large == async_doc_embeddings_large
        print("Async document embeddings (3-large, 512d) match sync.")

    except Exception as e:
        print(f"Error with text-embedding-3-large (possibly not available or API key issue): {e}")
        print("Skipping advanced example.")
    print("---")

if __name__ == "__main__":
    # Setup instructions:
    # 1. Install dependencies: pip install -e .[openai] python-dotenv
    # 2. Create a .env file in the root directory with: OPENAI_API_KEY=your_openai_api_key
    # 3. Run this script from the root directory: python examples/embeddings_openai.py
    asyncio.run(main()) 