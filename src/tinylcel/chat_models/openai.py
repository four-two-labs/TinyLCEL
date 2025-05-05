"""Implementation of a chat model using the OpenAI API."""

import os
from dataclasses import field
from dataclasses import dataclass
from typing import Any

import openai

from tinylcel.messages import AIMessage
from tinylcel.messages import BaseMessage
from tinylcel.messages import MessagesInput
from tinylcel.chat_models.base import BaseChatModel


def _get_openai_api_key(api_key: str | None) -> str:
    """Retrieve the OpenAI API key.

    Checks for the key in the provided argument first, then falls back to the
    `OPENAI_API_KEY` environment variable.

    Args:
        api_key: An optional API key provided directly.

    Returns:
        The resolved OpenAI API key.

    Raises:
        ValueError: If the API key is not provided via argument or environment
            variable.
    """
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError(
            'OPENAI_API_KEY environment variable not set or API key not provided.'
        )
    return key


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
        openai_api_key: Optional OpenAI API key. If not provided, the
            `OPENAI_API_KEY` environment variable will be used.

    Examples:
        >>> from tinylcel.chat_models.openai import ChatOpenAI
        >>> from tinylcel.messages import HumanMessage
        >>> # Assumes OPENAI_API_KEY environment variable is set
        >>> model = ChatOpenAI(model="gpt-4o-mini")
        >>> response = model.invoke([HumanMessage(content="Hello!")])
        >>> print(response.content)

        >>> # Chain with a parser
        >>> from tinylcel.output_parsers import StrOutputParser
        >>> chain = model | StrOutputParser()
        >>> result = chain.invoke([HumanMessage(content="Tell me a joke.")])
        >>> print(result)
    """
    model: str = 'gpt-3.5-turbo'
    temperature: float = 0.7
    max_tokens: int | None = None
    openai_api_key: str | None = field(default=None, repr=False)

    _client: openai.OpenAI = field(init=False, repr=False)
    _async_client: openai.AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self):
        """Initializes the synchronous and asynchronous OpenAI clients.

        Retrieves the API key using `_get_openai_api_key` and instantiates
        `openai.OpenAI` and `openai.AsyncOpenAI`.
        """
        resolved_api_key = _get_openai_api_key(self.openai_api_key)
        self._client = openai.OpenAI(api_key=resolved_api_key)
        self._async_client = openai.AsyncOpenAI(api_key=resolved_api_key)

    def _convert_message_to_dict(self, message: BaseMessage) -> dict[str, str]:
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
                role = message.role # Pass through unknown roles for now

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
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            api_kwargs["max_tokens"] = self.max_tokens

        response = self._client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        if choice.message.content is None:
             # Handle cases like function calls or empty responses if needed later
            raise ValueError("OpenAI response content is None")
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
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            api_kwargs["max_tokens"] = self.max_tokens

        response = await self._async_client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        if choice.message.content is None:
             # Handle cases like function calls or empty responses if needed later
            raise ValueError("OpenAI response content is None")
        return AIMessage(content=choice.message.content)
