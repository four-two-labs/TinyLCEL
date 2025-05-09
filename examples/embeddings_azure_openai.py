"""Example of using AzureOpenAIEmbeddings."""

import os
import asyncio
from dotenv import load_dotenv

from tinylcel.embeddings.openai import AzureOpenAIEmbeddings

async def main():
    load_dotenv()

    # Required environment variables for Azure OpenAI Embeddings
    api_key = os.environ.get('AZURE_OPENAI_APIKEY') or os.environ.get('OPENAI_API_KEY')
    azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    api_version = os.environ.get('AZURE_OPENAI_API_VERSION')
    azure_deployment = os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME') # Specific deployment for embeddings

    if not all([api_key, azure_endpoint, api_version, azure_deployment]):
        print("One or more required Azure OpenAI environment variables are missing.")
        print("Please set: AZURE_OPENAI_API_KEY (or OPENAI_API_KEY), AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        return

    print(f"--- Azure OpenAI Embeddings Example (Deployment: {azure_deployment}) ---")
    
    # You might want to specify dimensions if your Azure deployment supports it and you need a specific size
    # For example, if it's a text-embedding-3-small deployment:
    # embedder = AzureOpenAIEmbeddings(
    #     openai_api_key=api_key, # Can be passed explicitly or via env var
    #     azure_endpoint=azure_endpoint,
    #     api_version=api_version,
    #     azure_deployment=azure_deployment,
    #     dimensions=512 
    # )
    # For this general example, we'll use the deployment's default dimension.
    embedder = AzureOpenAIEmbeddings(
        dimensions=512,
        openai_api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=azure_deployment
    )

    query = "What is vector similarity search?"
    
    # --- Synchronous Usage ---
    print("\n--- Synchronous Usage ---")
    try:
        query_embedding = embedder.embed_query(query)
        print(f"Query: \"{query}\"")
        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"First 5 elements: {query_embedding[:5]}")
        print("---")

        doc_texts = [
            "Vector databases store and index vector embeddings for fast retrieval.",
            "Cosine similarity is a common metric for measuring similarity between two vectors.",
            "Azure OpenAI Service provides access to powerful language models."
        ]
        doc_embeddings = embedder.embed_documents(doc_texts)
        print(f"Documents to embed: {len(doc_texts)}")
        print(f"Number of embeddings generated: {len(doc_embeddings)}")
        if doc_embeddings:
            print(f"Dimension of first doc embedding: {len(doc_embeddings[0])}")
            print(f"First 5 elements of first doc embedding: {doc_embeddings[0][:5]}")
        print("---")

        # --- Asynchronous Usage --- #
        print("\n--- Asynchronous Usage ---")
        async_query_embedding = await embedder.aembed_query(query)
        assert query_embedding == async_query_embedding # Should be deterministic for same input
        print("Async query embedding matches sync embedding.")
        print(f"Async Embedding dimension: {len(async_query_embedding)}")
        print("---")

        async_doc_embeddings = await embedder.aembed_documents(doc_texts)
        assert doc_embeddings == async_doc_embeddings # Should match sync results
        print("Async document embeddings match sync embeddings.")
        if async_doc_embeddings:
            print(f"Async Dimension of first doc embedding: {len(async_doc_embeddings[0])}")
        print("---")

    except Exception as e:
        print(f"An error occurred during Azure OpenAI API interaction: {e}")
        print("Please check your Azure OpenAI service configuration and API key.")


if __name__ == "__main__":
    # Setup instructions:
    # 1. Install dependencies: pip install -e .[openai] python-dotenv
    # 2. Create a .env file in the project root with your Azure OpenAI credentials:
    #    AZURE_OPENAI_API_KEY=your_azure_api_key (or OPENAI_API_KEY if that's what you use for Azure)
    #    AZURE_OPENAI_ENDPOINT=your_azure_endpoint
    #    AZURE_OPENAI_API_VERSION=your_api_version (e.g., 2023-07-01-preview)
    #    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name
    # 3. Run this script from the project root directory: python examples/embeddings_azure_openai.py
    asyncio.run(main()) 