"""Implementation of embedding models using the Cohere API."""

from typing import Literal
from dataclasses import field
from dataclasses import dataclass

import cohere  # type: ignore[import-not-found]
from cohere.client import OMIT  # type: ignore[import-not-found]

from tinylcel.utils.auth import get_api_key
from tinylcel.embeddings.base import BaseEmbeddings


@dataclass
class CohereEmbeddings(BaseEmbeddings):
    """Cohere embedding model integration.

    This class provides an interface to Cohere's embedding models.
    It handles API key management and makes calls to the Cohere API
    to generate embeddings for documents and queries.

    Setup:
        Install the `cohere` package:
        ```bash
        pip install cohere
        ```
        And configure your API key by setting the `COHERE_API_KEY` environment
        variable, or by passing the `api_key` argument during instantiation.

    Attributes:
        model: Model name to use (e.g., "embed-english-v3.0").
        base_url: Optional custom base URL for the Cohere API.
        api_key: Optional Cohere API key.
        truncate: Truncation strategy ("NONE", "START", "END").
        dimensions: Optional dimension of the output embeddings. Defaults to the
            model's default if OMIT is used.

    Raises:
        ValueError: If the Cohere API key is not found.

    See Also:
        Cohere API documentation: https://docs.cohere.com/reference/embed
    """

    model: str = 'embed-english-v3.0'
    base_url: str | None = field(default=None, repr=True)
    api_key: str | None = field(default=None, repr=False)
    truncate: Literal['NONE', 'START', 'END'] = field(default='END', repr=True)
    dimensions: int = field(default=OMIT, repr=True)

    _client: cohere.ClientV2 = field(init=False, repr=False)
    _async_client: cohere.AsyncClientV2 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initializes the synchronous and asynchronous Cohere clients.

        Retrieves the API key and sets up the Cohere V2 clients. If a
        `base_url` is provided, it's used for client initialization.

        Raises:
            ValueError: If the Cohere API key cannot be resolved.
        """
        resolved_api_key = get_api_key(self.api_key, 'COHERE_API_KEY', 'Cohere')
        client_kwargs = {'api_key': resolved_api_key}
        async_client_kwargs = {'api_key': resolved_api_key}
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
            async_client_kwargs['base_url'] = self.base_url

        self._client = cohere.ClientV2(**client_kwargs)  # type: ignore[arg-type]
        self._async_client = cohere.AsyncClientV2(**async_client_kwargs)  # type: ignore[arg-type]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts.

        Args:
            texts: A list of document strings to embed.

        Returns:
            A list of embeddings, one for each input document.

        Raises:
            cohere.CohereError: If the Cohere API call fails.
        """
        response = self._client.embed(
            texts=texts,
            model=self.model,
            input_type='search_document',
            truncate=self.truncate,
            output_dimension=self.dimensions,
            embedding_types=['float'],
        )
        return response.embeddings.float  # type: ignore

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding for the query.

        Raises:
            cohere.CohereError: If the Cohere API call fails.
        """
        response = self._client.embed(
            texts=[text],
            model=self.model,
            input_type='search_query',
            truncate=self.truncate,
            output_dimension=self.dimensions,
            embedding_types=['float'],
        )
        return response.embeddings.float[0]  # type: ignore

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generate embeddings for a list of document texts.

        Args:
            texts: A list of document strings to embed.

        Returns:
            A list of embeddings, one for each input document.

        Raises:
            cohere.CohereError: If the Cohere API call fails.
        """
        response = await self._async_client.embed(
            texts=texts,
            model=self.model,
            input_type='search_document',
            embedding_types=['float'],
            output_dimension=self.dimensions,
            truncate=self.truncate,
        )
        return response.embeddings.float  # type: ignore

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously generate an embedding for a single query text.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding for the query.

        Raises:
            cohere.CohereError: If the Cohere API call fails.
        """
        response = await self._async_client.embed(
            texts=[text],
            model=self.model,
            input_type='search_query',
            embedding_types=['float'],
            output_dimension=self.dimensions,
            truncate=self.truncate,
        )
        return response.embeddings.float[0]  # type: ignore
