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
    """Abstract base class for chat models.

    Inherits from RunnableBase, expecting a list of BaseMessages as input
    and producing an AIMessage as output.
    """

    @abc.abstractmethod
    def _generate(self, messages: MessagesInput) -> AIMessage:
        """Synchronous generation method to be implemented by subclasses.

        Args:
            messages: The list of messages constituting the conversation history.

        Returns:
            An AIMessage containing the model's response.

        """
        ...

    @abc.abstractmethod
    async def _agenerate(self, messages: MessagesInput) -> AIMessage:
        """Asynchronous generation method to be implemented by subclasses.

        Args:
            messages: The list of messages constituting the conversation history.

        Returns:
            An awaitable resolving to an AIMessage containing the model's response.

        """
        ...

    def invoke(self, input: MessagesInput) -> AIMessage:
        """Synchronously invokes the chat model with a list of messages.

        Args:
            input: A list of BaseMessage objects.

        Returns:
            An AIMessage representing the model's response.

        """
        return self._generate(input)

    async def ainvoke(self, input: MessagesInput) -> AIMessage:
        """Asynchronously invokes the chat model with a list of messages.

        Args:
            input: A list of BaseMessage objects.

        Returns:
            An awaitable resolving to an AIMessage representing the model's response.

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
