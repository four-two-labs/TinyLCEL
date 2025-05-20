from dataclasses import field
from dataclasses import dataclass

from tinylcel.messages.base import BaseMessage
from tinylcel.messages.base import MessageContentBlock


@dataclass(frozen=True)
class BaseMessageChunk(BaseMessage):
    """Message chunk supporting concatenation of partial content.
    
    A BaseMessageChunk represents a portion of a message that can be concatenated with
    other chunks of the same role. The content can be either a single MessageContentBlock
    or a list of MessageContentBlocks.

    Attributes:
        content: The content of the message chunk, either as a single block or list of blocks
        role: The role of the message chunk (inherited from BaseMessage)
    """
    content: MessageContentBlock | list[MessageContentBlock]  # type: ignore

    def __add__(self, other: 'BaseMessageChunk') -> 'BaseMessageChunk':
        """Concatenate two message chunks of the same role.

        Args:
            other: Another message chunk to concatenate with this one

        Returns:
            A new message chunk containing the combined content

        Raises:
            TypeError: If other is not a BaseMessageChunk or has a different role,
                      or if content types are incompatible
        """
        if not isinstance(other, BaseMessageChunk) or self.role != other.role:
            raise TypeError(
                f'Cannot concatenate messages of different roles: '
                f'"{self.role}" and "{other.role}"'
            )

        match self.content, other.content:
            case dict() as a, dict() as b:
                return self.__class__(content=[a, b])
            case dict() as a, list() as b:
                return self.__class__(content=[a, *b])
            case list() as a, dict() as b:
                return self.__class__(content=[*a, b])
            case list() as a, list() as b:
                return self.__class__(content=[*a, *b])
            
        raise TypeError(
            f'Unsupported operand type(s) for +: '
            f'"{self.__class__.__name__}" and '
            f'"{other.__class__.__name__}"'
        )

@dataclass(frozen=True)
class HumanMessageChunk(BaseMessageChunk):
    """A chunk of a human message.
    
    Represents a portion of a message from a human user. The role is automatically
    set to "human" and cannot be changed.

    Attributes:
        role: Set to "human" (immutable)
        content: The content of the human message chunk
    """
    role: str = field(default="human", init=False)

@dataclass(frozen=True)
class AIMessageChunk(BaseMessageChunk):
    """A chunk of an AI message.
    
    Represents a portion of a message from an AI assistant. The role is automatically
    set to "ai" and cannot be changed.

    Attributes:
        role: Set to "ai" (immutable)
        content: The content of the AI message chunk
    """
    role: str = field(default="ai", init=False)

@dataclass(frozen=True)
class SystemMessageChunk(BaseMessageChunk):
    """A chunk of a system message.
    
    Represents a portion of a system message that provides context or instructions.
    The role is automatically set to "system" and cannot be changed.

    Attributes:
        role: Set to "system" (immutable)
        content: The content of the system message chunk
    """
    role: str = field(default="system", init=False)


def text_chunk(text: str) -> HumanMessageChunk:
    """Create a human message chunk containing text content.

    Args:
        text: The text content to include in the message chunk

    Returns:
        A HumanMessageChunk containing the text in a standard format
    """
    return HumanMessageChunk(content=[{"type": "text", "text": text}])

TextChunk = text_chunk
    
    
