"""Implementation of a chat model using the OpenAI API."""

from typing import Any
from typing import TypeVar
from dataclasses import field
from dataclasses import dataclass

import openai
from openai import NOT_GIVEN
from openai import AzureOpenAI
from openai._types import NotGiven
from openai import AsyncAzureOpenAI

from tinylcel.utils.auth import get_api_key
from tinylcel.messages.base import AIMessage
from tinylcel.messages.base import BaseMessage
from tinylcel.messages.base import MessagesInput
from tinylcel.chat_models.base import BaseChatModel
from tinylcel.messages.base import MessageContentBlock

# Added TypeVar for from_client method
T = TypeVar('T')


@dataclass
class ChatOpenAI(BaseChatModel):
    """
    A chat model that uses the OpenAI Chat Completion API.

    This class is a wrapper around OpenAI's API for models like GPT-3.5 Turbo,
    GPT-4, etc. It integrates with the TinyLCEL framework, allowing it to be
    used in runnable sequences.

    Authentication:
        Requires an OpenAI API key. This can be provided directly via the
        `openai_api_key` argument during initialization, or by setting the
        `OPENAI_API_KEY` environment variable.

    Dependencies:
        Requires the `openai` python package to be installed (`pip install openai`).

    Attributes:
        model: The name of the OpenAI model to use (e.g., "gpt-4o-mini",
            "gpt-3.5-turbo"). Defaults to "gpt-3.5-turbo".
        temperature: Sampling temperature for generation (0 to 2).
            Higher values make output more random, lower values make it more deterministic.
            Defaults to 0.7.
        max_tokens: Optional maximum number of tokens to generate.
            Defaults to None (no limit imposed by this class, but the model may
            have its own limits).
        api_key: Optional OpenAI API key. If not provided, the
            `OPENAI_API_KEY` environment variable will be used.
        base_url: Optional base URL for the OpenAI API.

    Examples:
        >>> from tinylcel.chat_models.openai import ChatOpenAI
        >>> from tinylcel.messages import HumanMessage
        >>> # Assumes OPENAI_API_KEY environment variable is set
        >>> model = ChatOpenAI(model='gpt-4o-mini')
        >>> response = model.invoke([HumanMessage(content='Hello!')])
        >>> print(response.content)

        >>> # Chain with a parser
        >>> from tinylcel.output_parsers import StrOutputParser
        >>> chain = model | StrOutputParser()
        >>> result = chain.invoke([HumanMessage(content='Tell me a joke.')])
        >>> print(result)
    """

    model: str = 'gpt-3.5-turbo'
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = field(default=None, repr=True)
    temperature: float | NotGiven = field(default=NOT_GIVEN, repr=True)
    max_tokens: int | NotGiven = field(default=NOT_GIVEN, repr=True)
    max_completion_tokens: int | NotGiven = field(default=NOT_GIVEN, repr=True)
    max_retries: int = field(default=openai.DEFAULT_MAX_RETRIES, repr=True)
    timeout: int | NotGiven = field(default=NOT_GIVEN, repr=True)

    _client: openai.OpenAI = field(init=True, repr=False, default=None)  # type: ignore[assignment]
    _async_client: openai.AsyncOpenAI = field(init=True, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initializes the synchronous and asynchronous OpenAI clients.

        Retrieves the API key and configures clients with retries, timeout, and base_url.

        Raises:
            ValueError: If the OpenAI API key cannot be resolved.
        """
        resolved_api_key = get_api_key(self.api_key, 'OPENAI_API_KEY', 'OpenAI')

        client_kwargs: dict[str, Any] = {
            'api_key': resolved_api_key,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
        }
        if self.base_url:
            client_kwargs['base_url'] = self.base_url

        self._client = self._client or openai.OpenAI(**client_kwargs)  # type: ignore[assignment]
        self._async_client = self._async_client or openai.AsyncOpenAI(**client_kwargs)  # type: ignore[assignment]

    def _convert_message_to_dict(self, message: BaseMessage) -> dict[str, str | list[MessageContentBlock]]:
        """Convert a TinyLCEL message to the OpenAI API dictionary format.

        Maps internal roles (`human`, `ai`, `system`) to the roles expected
        by the OpenAI API (`user`, `assistant`, `system`).

        Args:
            message: A `BaseMessage` (or subclass) instance.

        Returns:
            A dictionary with 'role' and 'content' keys formatted for the API.
        """
        match message.role:
            case 'human':
                role = 'user'
            case 'ai':
                role = 'assistant'
            case 'system':
                role = 'system'
            case _:
                # Or raise an error for unsupported roles?
                role = message.role  # Pass through unknown roles for now

        return {'role': role, 'content': message.content}

    def _generate(self, messages: MessagesInput) -> AIMessage:
        """Synchronous generation using the OpenAI Chat Completion API.

        Formats the input messages, calls the `create` method of the synchronous
        OpenAI client, and returns the response content as an `AIMessage`.

        Args:
            messages: A list of `BaseMessage` objects.

        Returns:
            An `AIMessage` containing the first choice's message content.

        Raises:
            ValueError: If the API response content is None.
            openai.APIError: If the OpenAI API returns an error.
        """
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        api_kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': message_dicts,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_completion_tokens': self.max_completion_tokens,
        }

        response = self._client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        if choice.message.content is None:
            raise ValueError('OpenAI response content is None')

        return AIMessage(content=choice.message.content)

    async def _agenerate(self, messages: MessagesInput) -> AIMessage:
        """Asynchronous generation using the OpenAI Chat Completion API.

        Formats the input messages, calls the `create` method of the asynchronous
        OpenAI client, and returns the response content as an `AIMessage`.

        Args:
            messages: A list of `BaseMessage` objects.

        Returns:
            An awaitable resolving to an `AIMessage` containing the first choice's
            message content.

        Raises:
            ValueError: If the API response content is None.
            openai.APIError: If the OpenAI API returns an error.
        """
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        api_kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': message_dicts,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_completion_tokens': self.max_completion_tokens,
        }

        response = await self._async_client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        if choice.message.content is None:
            raise ValueError('OpenAI response content is None')

        return AIMessage(content=choice.message.content)


@dataclass
class AzureChatOpenAI(ChatOpenAI):
    """
    A chat model that uses the Azure OpenAI Service.

    This class inherits from `ChatOpenAI` and adapts it for use with
    Azure OpenAI deployments.

    Authentication:
        Requires an Azure OpenAI API key, an endpoint, and an API version.
        The API key can be provided via `openai_api_key` or the
        `OPENAI_API_KEY` (or `AZURE_OPENAI_API_KEY`) environment variable.
        The endpoint and API version must be provided.

    Attributes:
        azure_endpoint: The endpoint URL for your Azure OpenAI resource.
        api_version: The API version for the Azure OpenAI service
            (e.g., "2023-07-01-preview").
        azure_deployment: The name of your Azure OpenAI deployment. This will be
            used as the `model` parameter in API calls.
        model: This field is inherited from ChatOpenAI. For Azure, it's
            recommended to set this to your `azure_deployment` name.
            If `azure_deployment` is provided, `model` will be overridden
            by `azure_deployment` in API calls.
        temperature: Sampling temperature (0 to 2). Defaults to 0.7.
        max_tokens: Optional maximum number of tokens. Defaults to None.
        openai_api_key: Optional Azure OpenAI API key.

    Examples:
        >>> from tinylcel.chat_models.openai import AzureChatOpenAI
        >>> from tinylcel.messages import HumanMessage
        >>> # Assumes AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        >>> # AZURE_OPENAI_DEPLOYMENT_NAME and AZURE_OPENAI_API_VERSION
        >>> # environment variables are set, or values are passed directly.
        >>> model = AzureChatOpenAI(
        ...     azure_endpoint='YOUR_AZURE_ENDPOINT',
        ...     api_version='YOUR_API_VERSION',
        ...     azure_deployment='your-deployment-name',
        ...     model='your-deployment-name',  # Often same as azure_deployment
        ... )
        >>> response = model.invoke([HumanMessage(content='Hello Azure OpenAI!')])
        >>> print(response.content)
    """

    azure_endpoint: str = field(default='', repr=True)
    api_version: str | None = field(default=None, repr=True)
    azure_deployment: str | None = field(default=None, repr=True)

    _client: AzureOpenAI = field(init=True, repr=False, default=None)  # type: ignore
    _async_client: AsyncAzureOpenAI = field(init=True, repr=False, default=None)  # type: ignore

    def __post_init__(self) -> None:
        """Initializes the Azure OpenAI clients."""
        resolved_api_key = get_api_key(self.api_key, 'AZURE_OPENAI_API_KEY', 'Azure OpenAI')
        client_kwargs_azure: dict[str, Any] = {
            'api_key': resolved_api_key,
            'azure_endpoint': self.azure_endpoint,
            'api_version': self.api_version,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
        }
        self._client = self._client or AzureOpenAI(**client_kwargs_azure)  # type: ignore
        self._async_client = self._async_client or AsyncAzureOpenAI(**client_kwargs_azure)  # type: ignore
        self.model = self.model or self.azure_deployment  # type: ignore


def from_client(
    client: openai.OpenAI,
    async_client: openai.AsyncOpenAI,
    model: str,
    temperature: float | NotGiven = NOT_GIVEN,
    max_tokens: int | NotGiven = NOT_GIVEN,
    max_completion_tokens: int | NotGiven = NOT_GIVEN,
    max_retries: int = openai.DEFAULT_MAX_RETRIES,
    timeout: int | NotGiven = NOT_GIVEN,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance from pre-configured clients."""
    return ChatOpenAI(
        model=model,
        api_key='not_used_with_from_client',
        base_url=None,
        temperature=temperature,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        max_retries=max_retries,
        timeout=timeout,
        _client=client.copy(timeout=timeout, max_retries=max_retries),
        _async_client=async_client.copy(timeout=timeout, max_retries=max_retries),
    )


def from_azure_client(
    client: openai.AzureOpenAI,
    async_client: openai.AsyncAzureOpenAI,
    model: str,
    temperature: float | NotGiven = NOT_GIVEN,
    max_tokens: int | NotGiven = NOT_GIVEN,
    max_completion_tokens: int | NotGiven = NOT_GIVEN,
    max_retries: int = openai.DEFAULT_MAX_RETRIES,
    timeout: int | NotGiven = NOT_GIVEN,
) -> AzureChatOpenAI:
    """Create an AzureChatOpenAI instance from pre-configured Azure clients."""
    return AzureChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        max_retries=max_retries,
        timeout=timeout,
        api_key='not_used_with_from_azure_client',
        azure_endpoint='placeholder_from_azure_client',
        api_version='placeholder_from_azure_client',
        azure_deployment='placeholder_from_azure_client',
        _client=client.copy(timeout=timeout, max_retries=max_retries),
        _async_client=async_client.copy(timeout=timeout, max_retries=max_retries),
    )
