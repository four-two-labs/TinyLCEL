"""OpenAI chat model implementation for TinyLCEL."""

import os
from dataclasses import field
from dataclasses import dataclass

import openai

from tinylcel.messages import AIMessage
from tinylcel.messages import BaseMessage
from tinylcel.messages import MessagesInput
from tinylcel.chat_models.base import BaseChatModel


def _get_openai_api_key(api_key: str | None) -> str:
    """Get the OpenAI API key from arg or environment variable."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set or API key not provided."
        )
    return key


@dataclass
class ChatOpenAI(BaseChatModel):
    """Wrapper around OpenAI Chat Completion API models.

    Requires the 'openai' package and an API key set via the
    OPENAI_API_KEY environment variable or passed as 'openai_api_key'.
    """

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    openai_api_key: str | None = field(default=None, repr=False)

    _client: openai.OpenAI = field(init=False, repr=False)
    _async_client: openai.AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the OpenAI clients."""
        resolved_api_key = _get_openai_api_key(self.openai_api_key)
        self._client = openai.OpenAI(api_key=resolved_api_key)
        self._async_client = openai.AsyncOpenAI(api_key=resolved_api_key)

    def _convert_message_to_dict(self, message: BaseMessage) -> dict[str, str]:
        """Convert a BaseMessage to the OpenAI API dictionary format."""
        return {"role": message.role, "content": message.content}

    def _generate(self, messages: MessagesInput) -> AIMessage:
        """Synchronous generation using the OpenAI API."""
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=message_dicts, # type: ignore
            temperature=self.temperature,
        )
        choice = response.choices[0]
        if choice.message.content is None:
             # Handle cases like function calls or empty responses if needed later
            raise ValueError("OpenAI response content is None")
        return AIMessage(content=choice.message.content)

    async def _agenerate(self, messages: MessagesInput) -> AIMessage:
        """Asynchronous generation using the OpenAI API."""
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=message_dicts, # type: ignore
            temperature=self.temperature,
        )
        choice = response.choices[0]
        if choice.message.content is None:
             # Handle cases like function calls or empty responses if needed later
            raise ValueError("OpenAI response content is None")
        return AIMessage(content=choice.message.content)
