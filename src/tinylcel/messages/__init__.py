from tinylcel.messages.base import AIMessage
from tinylcel.messages.base import BaseMessage
from tinylcel.messages.base import HumanMessage
from tinylcel.messages.base import SystemMessage
from tinylcel.messages.chunks import AIMessageChunk
from tinylcel.messages.chunks import BaseMessageChunk
from tinylcel.messages.chunks import HumanMessageChunk
from tinylcel.messages.chunks import SystemMessageChunk

type MessagesInput = list[BaseMessage]

__all__ = [
    'AIMessage',
    'AIMessageChunk',
    'BaseMessage',
    'BaseMessageChunk',
    'HumanMessage',
    'HumanMessageChunk',
    'MessagesInput',
    'SystemMessage',
    'SystemMessageChunk',
]
