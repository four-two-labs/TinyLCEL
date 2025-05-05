"""Base classes for chat models and messages in TinyLCEL."""

import abc
from dataclasses import dataclass, field

from tinylcel.messages import AIMessage, MessagesInput
from tinylcel.runnable import RunnableBase


@dataclass(frozen=True)
class BaseMessage:
    """Base class for messages in a chat conversation."""

    content: str
    role: str = field(init=False) # Role is determined by subclass


@dataclass(frozen=True)
class HumanMessage(BaseMessage):
    """A message from the human user."""

    role: str = field(default="human", init=False)


@dataclass(frozen=True)
class SystemMessage(BaseMessage):
    """A message setting the context for the AI assistant (system prompt)."""

    role: str = field(default="system", init=False)


class BaseChatModel(RunnableBase[MessagesInput, AIMessage], abc.ABC):
    """
    Abstract base class for language models that expose a chat interface.

    This class provides the core functionality for chat models within the
    Tiny LangChain Expression Language (TinyLCEL) framework. It inherits
    from `RunnableBase`, defining the input type as a list of `BaseMessage`
    objects (`MessagesInput`) and the output type as a single `AIMessage`.

    Subclasses must implement the `_generate` and `_agenerate` methods
    to provide the specific logic for interacting with the underlying
    language model API synchronously and asynchronously, respectively.

    The `invoke` and `ainvoke` methods are implemented here, calling the
    respective `_generate` and `_agenerate` methods.
    """

    @abc.abstractmethod
    def _generate(self, messages: MessagesInput) -> AIMessage:
        """
        Synchronous generation method.

        This method should be implemented by subclasses to handle the synchronous
        call to the underlying chat model API.

        Args:
            messages: A list of `BaseMessage` objects representing the
                conversation history and prompt.

        Returns:
            An `AIMessage` containing the model's response.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            Exception: Any exception raised by the underlying API call.
        """
        ...

    @abc.abstractmethod
    async def _agenerate(self, messages: MessagesInput) -> AIMessage:
        """
        Asynchronous generation method.

        This method should be implemented by subclasses to handle the asynchronous
        call to the underlying chat model API.

        Args:
            messages: A list of `BaseMessage` objects representing the
                conversation history and prompt.

        Returns:
            An awaitable resolving to an `AIMessage` containing the model's response.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            Exception: Any exception raised by the underlying API call.
        """
        ...

    def invoke(self, input: MessagesInput) -> AIMessage:
        """
        Invoke the chat model synchronously.

        This method takes a list of messages, calls the `_generate` method
        implemented by the subclass, and returns the resulting AI message.

        Args:
            input: A list of `BaseMessage` objects.

        Returns:
            An `AIMessage` representing the model's response.
        """
        return self._generate(input)

    async def ainvoke(self, input: MessagesInput) -> AIMessage:
        """
        Invoke the chat model asynchronously.

        This method takes a list of messages, calls the `_agenerate` method
        implemented by the subclass, and returns the resulting AI message.

        Args:
            input: A list of `BaseMessage` objects.

        Returns:
            An awaitable resolving to an `AIMessage` representing the model's
            response.
        """
        return await self._agenerate(input)

# Potential future addition: ChatResult for more complex outputs
# @dataclass
# class ChatGeneration:
#     message: AIMessage
#     generation_info: dict | None = None

# @dataclass
# class ChatResult:
#     generations: list[ChatGeneration]
#     llm_output: dict | None = None
