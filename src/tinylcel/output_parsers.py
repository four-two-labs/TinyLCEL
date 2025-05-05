"""Output parsers for TinyLCEL."""

import abc
from typing import Any
from typing import Protocol
from dataclasses import dataclass
from typing import runtime_checkable

from tinylcel.runnable import RunnableBase


@runtime_checkable
class HasContent(Protocol):
    content: Any

@runtime_checkable
class HasStringContent(Protocol):
    content: str


@dataclass
class BaseOutputParser[OutputType](RunnableBase[Any, OutputType], abc.ABC):
    """
    Abstract base class for parsing the output of an LLM call.

    Takes the raw output of an LLM (Any) and parses it into a
    more structured format (OutputType).
    """

    @abc.abstractmethod
    def parse(self, output: Any) -> OutputType:
        """
        Parse the raw output of an LLM into the desired format.

        Args:
            output: The raw output from the language model.

        Returns:
            The parsed output.
        """
        ...

    async def aparse(self, output: Any) -> OutputType:
        """
        Asynchronously parse the raw output of an LLM.
        Default implementation runs the sync `parse` method in a thread pool.

        Args:
            output: The raw output from the language model.

        Returns:
            An awaitable resolving to the parsed output.
        """
        return self.parse(output)

    # Inherited invoke/ainvoke will call parse/aparse
    def invoke(self, input: Any) -> OutputType:
        return self.parse(input)

    async def ainvoke(self, input: Any) -> OutputType:
        return await self.aparse(input)


@dataclass
class StrOutputParser(BaseOutputParser[str]):
    """
    Parses the output of an LLM call that is expected to be a message
    with string content (like AIMessage) into a simple string.
    """

    def parse(self, output: HasContent | HasStringContent | str) -> str:
        """
        Extracts the string content from the input.

        Args:
            output: An object assumed to have a 'content' attribute (e.g., BaseMessage).

        Returns:
            The string content.

        Raises:
            TypeError: If the output is not a BaseMessage or doesn't have 'content'.
            ValueError: If the content is not a string.
        """
        content: Any | None = None
        match output:
            case HasStringContent(content=content): ...                            
            case HasContent(content=content): ...
            case str() as content: ...            
            case _:
                raise TypeError(
                    f"Expected object with 'content' attribute, got {type(output)}"
                )
        
        if isinstance(content, str):
            return content

        raise ValueError(f"Expected string content, got {type(content)}")

