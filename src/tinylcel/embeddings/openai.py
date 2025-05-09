"""Implementation of embedding models using the OpenAI API."""

from typing import Any
from dataclasses import field
from dataclasses import dataclass
from dataclasses import KW_ONLY

import openai
from openai import Timeout
from openai import AzureOpenAI
from openai import AsyncAzureOpenAI
from openai._types import NOT_GIVEN
from openai._types import NotGiven

from tinylcel.embeddings.base import BaseEmbeddings
from tinylcel.chat_models.openai import _get_openai_api_key


@dataclass
class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embedding model integration.

    This class provides an interface to generate embeddings using OpenAI's
    various embedding models (e.g., "text-embedding-ada-002", 
    "text-embedding-3-small", "text-embedding-3-large").

    Attributes:
        model: The name of the OpenAI embedding model to use.
        openai_api_key: Optional OpenAI API key. If not provided, the client
            will attempt to use the `OPENAI_API_KEY` environment variable.
        base_url: Optional custom base URL for the OpenAI API.
        dimensions: Optional. The number of dimensions the resulting output 
            embeddings should have. Only supported for some newer models.
        max_retries: Maximum number of retries for API requests.
        timeout: Timeout configuration for API requests. Can be a float (seconds),
            an `openai.Timeout` object, or `NOT_GIVEN` to use client defaults.

    Raises:
        ValueError: If the OpenAI API key is not provided or found in the environment.

    Examples:
        >>> from tinylcel.embeddings.openai import OpenAIEmbeddings
        >>> # Assumes OPENAI_API_KEY environment variable is set
        >>> embedder = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=512)
        >>> query_embedding = embedder.embed_query('Hello, world!')
        >>> print(len(query_embedding))
        512
        >>> documents = ['This is document one.', 'This is document two.']
        >>> doc_embeddings = embedder.embed_documents(documents)
        >>> print(len(doc_embeddings))
        2
        >>> print(len(doc_embeddings[0]))
        512
    """
    model: str = 'text-embedding-ada-002'
    openai_api_key: str | None = field(default=None, repr=False)
    base_url: str | None = field(default=None, repr=True)
    dimensions: int | None = field(default=None, repr=True)
    max_retries: int = field(default=openai.DEFAULT_MAX_RETRIES, repr=True)
    timeout: float | Timeout | NotGiven = field(default=NOT_GIVEN, repr=True)

    _client: openai.OpenAI = field(init=False, repr=False)
    _async_client: openai.AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self):
        """Initializes the synchronous and asynchronous OpenAI clients."""
        resolved_api_key = _get_openai_api_key(self.openai_api_key)
        
        client_kwargs = {
            'api_key': resolved_api_key,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }
        if self.base_url:
            client_kwargs['base_url'] = self.base_url

        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)

    def _prepare_create_embedding_kwargs(self, texts: list[str]) -> dict[str, Any]:
        """Prepares the keyword arguments for the OpenAI embeddings.create call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A dictionary of keyword arguments for the API call.
        """
        kwargs: dict[str, Any] = {
            'model': self.model,
            'input': texts,
        }
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return kwargs

    def _embed_with_client(self, client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
        """Generates embeddings using the synchronous OpenAI client.

        Args:
            client: The synchronous `openai.OpenAI` client instance.
            texts: A list of texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        api_kwargs = self._prepare_create_embedding_kwargs(texts)
        response = client.embeddings.create(**api_kwargs) # type: ignore
        return [item.embedding for item in response.data]

    async def _aembed_with_client(self, client: openai.AsyncOpenAI, texts: list[str]) -> list[list[float]]:
        """Generates embeddings using the asynchronous OpenAI client.

        Args:
            client: The asynchronous `openai.AsyncOpenAI` client instance.
            texts: A list of texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        api_kwargs = self._prepare_create_embedding_kwargs(texts)
        response = await client.embeddings.create(**api_kwargs) # type: ignore
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of document texts.

        Args:
            texts: A list of document strings to embed.

        Returns:
            A list of embeddings, one for each input document.
        """
        return self._embed_with_client(self._client, texts)

    def embed_query(self, text: str) -> list[float]:
        """Generates an embedding for a single query text.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding for the query.
        """
        embeddings = self._embed_with_client(self._client, [text])
        return embeddings[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generates embeddings for a list of document texts.

        Args:
            texts: A list of document strings to embed.

        Returns:
            A list of embeddings, one for each input document.
        """
        return await self._aembed_with_client(self._async_client, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously generates an embedding for a single query text.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding for the query.
        """
        embeddings = await self._aembed_with_client(self._async_client, [text])
        return embeddings[0]

@dataclass
class AzureOpenAIEmbeddings(OpenAIEmbeddings):
    """Azure OpenAI Service embedding model integration.

    This class adapts `OpenAIEmbeddings` for use with Azure OpenAI deployments.

    Key init args (in addition to those inherited from OpenAIEmbeddings):
        azure_endpoint: str. The endpoint URL for your Azure OpenAI resource.
        api_version: str. The API version for the Azure OpenAI service (e.g., "2023-07-01-preview").
        azure_deployment: str. The name of your Azure OpenAI deployment. This will be used as the `model`.
    
    All other parameters from `OpenAIEmbeddings` (like `model`, `dimensions`, 
    `openai_api_key`, `max_retries`, `timeout`) are also available. If `azure_deployment`
    is provided, it will override the `model` attribute.

    Example:
        >>> from tinylcel.embeddings.openai import AzureOpenAIEmbeddings
        >>> # Assumes AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc. are set
        >>> embedder = AzureOpenAIEmbeddings(
        ...     azure_endpoint="YOUR_ENDPOINT", 
        ...     api_version="YOUR_API_VERSION", 
        ...     azure_deployment="your-text-embedding-deployment",
        ...     dimensions=512 # If your deployment/model supports it
        ... )
        >>> query_embedding = embedder.embed_query('Test query for Azure')
        >>> print(len(query_embedding))
        512
    """
    _: KW_ONLY # Makes subsequent fields keyword-only
    azure_endpoint: str
    api_version: str
    azure_deployment: str | None = None # Deployment name often acts as model name

    # Override client types to be Azure specific
    _client: AzureOpenAI = field(init=False, repr=False)
    _async_client: AsyncAzureOpenAI = field(init=False, repr=False)

    def __post_init__(self):
        """Initializes the Azure OpenAI clients.
        
        Note: Calls super().__post_init__() to ensure parent class
        attributes like openai_api_key are processed if needed, though here
        we re-initialize clients entirely for Azure.
        """
        # Though OpenAIEmbeddings.__post_init__ initializes clients, 
        # we re-initialize them here specifically for Azure.
        # The `openai_api_key` (and its resolution via _get_openai_api_key) 
        # and other config like `max_retries`, `timeout` are inherited.
        
        resolved_api_key = _get_openai_api_key(self.openai_api_key)

        client_kwargs: dict[str, Any] = {
            'api_key': resolved_api_key,
            'azure_endpoint': self.azure_endpoint,
            'api_version': self.api_version,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }
        # base_url is not typically used with AzureOpenAI client when azure_endpoint is set.
        # If self.base_url was intended for Azure, it might need specific handling or 
        # the AzureOpenAI client might not support it in the same way.
        # For now, not passing self.base_url to Azure clients.

        self._client = AzureOpenAI(**client_kwargs)
        self._async_client = AsyncAzureOpenAI(**client_kwargs)

        # If azure_deployment is set, it should be used as the model name for API calls.
        # The inherited _prepare_create_embedding_kwargs uses self.model.
        if self.azure_deployment:
            self.model = self.azure_deployment
        elif not self.model:
            raise ValueError("Either 'azure_deployment' or a valid 'model' name must be provided for Azure OpenAI embeddings.")

