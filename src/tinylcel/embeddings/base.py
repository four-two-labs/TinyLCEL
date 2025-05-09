"""Base interface for embedding models."""

import asyncio
from abc import ABC
from abc import abstractmethod


class BaseEmbeddings(ABC):
    """
    Base interface for embedding models.

    This class defines the common methods that all embedding model integrations
    in TinyLCEL should implement. It provides a standardized way to generate
    embeddings for single queries and batches of documents.

    Subclasses should implement the synchronous `embed_documents` and
    `embed_query` methods. Basic asynchronous versions are provided by default,
    running the synchronous methods in a thread pool.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: A list of document strings to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            The order of embeddings in the output list corresponds to the
            order of texts in the input list.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding for the query.
        """
        pass

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously generate embeddings for a list of documents.

        This default implementation runs the synchronous `embed_documents`
        method in a separate thread. Override this method for native 
        async performance.

        Args:
            texts: A list of document strings to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            The order of embeddings in the output list corresponds to the
            order of texts in the input list.
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """
        Asynchronously generate an embedding for a single query text.

        This default implementation runs the synchronous `embed_query`
        method in a separate thread. Override this method for native 
        async performance.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding for the query.
        """
        return await asyncio.to_thread(self.embed_query, text)
