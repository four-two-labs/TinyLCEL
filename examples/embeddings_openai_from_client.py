"""Example of using standalone from_client() function."""

import asyncio
from typing import List

import openai
from dotenv import load_dotenv

from tinylcel.providers.openai.embeddings import from_client


async def main() -> None:
    load_dotenv()
    print("--- Example: from_client() ---")

    try:
        # 1. Create pre-configured OpenAI clients
        original_sync_client = openai.OpenAI()
        original_async_client = openai.AsyncOpenAI()
        print("Successfully initialized original OpenAI clients for embeddings.")

        # 2. Instantiate OpenAIEmbeddings using standalone from_client function
        embedding_model = from_client(
            client=original_sync_client,
            async_client=original_async_client,
            model='text-embedding-3-small',
            dimensions=256
        )
        print(f"OpenAIEmbeddings instance created for model: {embedding_model.model}")

        # 3. Use the embedding model
        documents_to_embed: List[str] = [
            "TinyLCEL facilitates working with LangChain patterns.",
            "The from_client pattern is useful for custom client configurations."
        ]
        query_to_embed: str = "How to use from_client for OpenAIEmbeddings?"

        print(f"\nEmbedding {len(documents_to_embed)} documents...")
        doc_embeddings = embedding_model.embed_documents(documents_to_embed)
        print(f"Embedded {len(doc_embeddings)} documents.")
        if doc_embeddings:
            print(f"Dimension of first document embedding: {len(doc_embeddings[0])}")

        print(f"\nEmbedding query: '{query_to_embed}'...")
        query_embedding = embedding_model.embed_query(query_to_embed)
        print(f"Embedded query. Dimension: {len(query_embedding)}")

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        print("Ensure your OPENAI_API_KEY is set and valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'original_async_client' in locals() and original_async_client:
            await original_async_client.close()
        if 'original_sync_client' in locals() and original_sync_client:
            original_sync_client.close()
        print("\nOriginal OpenAI embedding clients closed if initialized.")


if __name__ == "__main__":
    asyncio.run(main()) 