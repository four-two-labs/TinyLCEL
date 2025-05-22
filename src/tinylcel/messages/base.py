"""Message types for TinyLCEL chat models."""

from typing import Mapping
from dataclasses import field
from dataclasses import dataclass


type MessageContentBlock = Mapping[str, str | Mapping[str, str]]


@dataclass(frozen=True)
class BaseMessage:
    """Base class for messages in a chat conversation."""
    content: str | list[MessageContentBlock]
    role: str = field(init=False)


@dataclass(frozen=True)
class HumanMessage(BaseMessage):
    """A message from the human user."""

    role: str = field(default="human", init=False)


@dataclass(frozen=True)
class AIMessage(BaseMessage):
    """A message from the AI assistant."""

    role: str = field(default="ai", init=False)


@dataclass(frozen=True)
class SystemMessage(BaseMessage):
    """A message setting the context for the AI assistant (system prompt)."""

    role: str = field(default="system", init=False)



type MessagesInput = list[BaseMessage]



# Message chunk support (minimal)
@dataclass(frozen=True)
class BaseMessageChunk(BaseMessage):
    """Message chunk supporting concatenation of partial content."""
    content: MessageContentBlock | list[MessageContentBlock]  # type: ignore

    def __add__(self, other: 'BaseMessageChunk') -> 'BaseMessageChunk':
        match self.content, other.content:
            case dict() as a, dict() as b:
                return self.__class__(content=[a, b])
            case dict() as a, list(dict()) as b:
                return self.__class__(content=[a, *b])
            case list(dict()) as a, dict() as b:
                return self.__class__(content=[*a, b])
            
        raise TypeError(
            f'unsupported operand type(s) for +: '
            f'"{self.__class__.__name__}" and '
            f'"{other.__class__.__name__}"'
        )

@dataclass(frozen=True)
class HumanMessageChunk(BaseMessageChunk):
    """A chunk of a human message."""
    role: str = field(default="human", init=False)

@dataclass(frozen=True)
class AIMessageChunk(BaseMessageChunk):
    """A chunk of an AI message."""
    role: str = field(default="ai", init=False)

@dataclass(frozen=True)
class SystemMessageChunk(BaseMessageChunk):
    """A chunk of a system message."""
    role: str = field(default="system", init=False)
