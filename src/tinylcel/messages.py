"""Message types for TinyLCEL chat models."""

from dataclasses import field
from dataclasses import dataclass


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
class AIMessage(BaseMessage):
    """A message from the AI assistant."""

    role: str = field(default="ai", init=False)


@dataclass(frozen=True)
class SystemMessage(BaseMessage):
    """A message setting the context for the AI assistant (system prompt)."""

    role: str = field(default="system", init=False)


# Type alias for a list of messages, representing a conversation history
type MessagesInput = list[BaseMessage]
