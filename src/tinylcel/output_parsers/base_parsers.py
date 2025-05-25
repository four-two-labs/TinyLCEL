"""Output parsers for TinyLCEL."""

import re
import abc
import json
from typing import Any
from typing import Literal
from typing import Protocol
from dataclasses import field
from dataclasses import dataclass
from typing import runtime_checkable

import yaml

from tinylcel.runnable import RunnableBase


@runtime_checkable
class HasContent(Protocol):
    """Protocol for objects that have a content attribute."""

    content: Any


class ParseError(Exception):
    """Base exception for output parsing errors."""

    def __init__(self, message: str, original_text: str | None = None) -> None:
        """Initialize ParseError with message and optional original text.

        Args:
            message: Error message describing the parsing failure.
            original_text: The original text that failed to parse, if available.
        """
        super().__init__(message)
        self.original_text = original_text


@dataclass(frozen=True)
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
        """Asynchronously parse the raw output of an LLM.

        Default implementation runs the sync `parse` method in a thread pool.

        Args:
            output: The raw output from the language model.

        Returns:
            An awaitable resolving to the parsed output.
        """
        return self.parse(output)

    def invoke(self, input: Any) -> OutputType:
        """Synchronous invocation."""
        return self.parse(input)

    async def ainvoke(self, input: Any) -> OutputType:
        """Asynchronous invocation."""
        return await self.aparse(input)


@dataclass(frozen=True)
class StrOutputParser(BaseOutputParser[str]):
    """Parses the output of an LLM call that is expected to be a message.

    Converts message with string content (like AIMessage) into a simple string.
    """

    def parse(self, input: HasContent | str) -> str:
        """
        Extracts the string content from the input.

        Args:
            input: An object assumed to have a 'content' attribute (e.g., BaseMessage).

        Returns:
            The string content.

        Raises:
            TypeError: If the input is not a BaseMessage or doesn't have 'content'.
            ValueError: If the content is not a string.
        """
        content: Any | None = None
        match input:
            case HasContent(content=content):
                ...
            case str() as content:
                ...
            case _:
                raise TypeError(f"Expected object with 'content' attribute, got {type(input)}")

        if isinstance(content, str):
            return content

        raise ValueError(f'Expected string content, got {type(content)}')


def FenceOutputParser(  # noqa: N802, C901
    fence_type: str = '', mode: Literal['relaxed', 'strict', 'typed_strict'] = 'relaxed'
) -> BaseOutputParser[str]:
    r"""
    Factory function to create a fence output parser for a specific language.

    Creates a parser that extracts content from markdown code fences with an optional
    language tag. The parser handles various fence formats including missing closing
    fences, extra whitespace, and fences without language tags.

    Args:
        fence_type: The language tag to look for (e.g., 'python', 'json', 'sql').
                   If empty, matches any language tag or no tag.
        mode: Parsing mode that controls strictness:
              - 'relaxed': Tries specific fence, then generic fence, then unfenced content (default)
              - 'strict': Requires fenced content, but accepts any language tag
              - 'typed_strict': Requires fenced content with the correct language tag

    Returns:
        A FenceOutputParser instance configured for the specified fence type and mode.

    Raises:
        ValueError: If an invalid mode is provided.

    Example:
        >>> python_parser = FenceOutputParser('python', mode='typed_strict')
        >>> result = python_parser.parse(AIMessage(content='```python\\nprint("hello")\\n```'))
        >>> print(result)  # 'print("hello")'

        >>> # This would raise ParseError in typed_strict mode:
        >>> result = python_parser.parse(AIMessage(content='print("hello")'))
    """
    valid_modes = {'relaxed', 'strict', 'typed_strict'}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}")

    # Pre-compile regexes in factory
    specific_regex = (
        re.compile(rf'```{re.escape(fence_type)}(?:\n(.*?)(?:\n```|$)|```)', re.IGNORECASE | re.DOTALL)
        if fence_type
        else None
    )

    generic_regex = re.compile(r'```(?:[^\n]*)?(?:\n(.*?)(?:\n```|$)|```)', re.IGNORECASE | re.DOTALL)

    def process_matched_content(content: str | None) -> str:
        """Process matched content from regex groups."""
        if content is not None:
            stripped_content = content.strip()
            return '' if stripped_content == '```' else stripped_content
        return ''

    def parse_logic(text: str) -> str:
        """Core parsing logic configured by factory parameters."""
        if fence_type and specific_regex:
            specific_match = specific_regex.search(text)
            if specific_match:
                return process_matched_content(specific_match.group(1))

            if mode == 'typed_strict':
                raise ParseError(f"No fenced block with language '{fence_type}' found in text", original_text=text)

            if mode == 'strict':
                generic_match = generic_regex.search(text)
                if generic_match:
                    return process_matched_content(generic_match.group(1))
                raise ParseError(f'No fenced content found in {mode} mode', original_text=text)

            # Enhanced relaxed mode: try generic fence before falling back to whole content
            generic_match = generic_regex.search(text)
            if generic_match:
                return process_matched_content(generic_match.group(1))

            return text.strip()

        generic_match = generic_regex.search(text)
        if generic_match:
            return process_matched_content(generic_match.group(1))

        if mode in {'strict', 'typed_strict'}:
            raise ParseError(f'No fenced content found in {mode} mode', original_text=text)

        return text.strip()

    @dataclass(frozen=True)
    class _FencedOutputParser(BaseOutputParser[str]):
        """Parses fenced output from an LLM call."""

        _parser: StrOutputParser = field(default_factory=StrOutputParser)

        def parse(self, input: Any) -> str:
            """Parse fenced content from the input based on the configured mode."""
            text = self._parser.parse(input)
            return parse_logic(text)

    return _FencedOutputParser()


@dataclass(frozen=True)
class JsonOutputParser(BaseOutputParser[Any]):
    """
    Parses JSON output from an LLM call.

    Handles potential markdown code fences (e.g., ```json ... ```) and
    leading/trailing whitespace. Uses FencedOutputParser internally for
    consistent fence handling.
    """

    _fenced_parser: BaseOutputParser[str] = field(default_factory=lambda: FenceOutputParser('json', mode='relaxed'))

    def parse(self, input: Any) -> Any:
        """
        Parses the string input into a Python object via JSON.

        Args:
            input: An object assumed to have a 'content' attribute (e.g., AIMessage).

        Returns:
            The parsed Python object (dict, list, str, int, float, bool, None).

        Raises:
            ParseError: If the input cannot be parsed as a string or if the
                extracted content is invalid JSON.
        """
        json_string = self._fenced_parser.parse(input)
        if not json_string:
            return {}

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ParseError(
                f'Failed to parse extracted string as JSON: {e}. String was: {json_string!r}',
                original_text=StrOutputParser().parse(input),
            ) from e


@dataclass(frozen=True)
class YamlOutputParser(BaseOutputParser[Any]):
    """
    Parses YAML output from an LLM call.

    Handles potential markdown code fences (e.g., ```yaml ... ```) and
    leading/trailing whitespace. Uses FencedOutputParser internally for
    consistent fence handling. Uses yaml.safe_load to prevent arbitrary code execution.
    """

    _fenced_parser: BaseOutputParser[str] = field(default_factory=lambda: FenceOutputParser('yaml', mode='relaxed'))

    def parse(self, input: Any) -> Any:
        """
        Parses the string input as YAML after extracting content.

        Args:
            input: An object assumed to have a 'content' attribute (e.g., AIMessage).

        Returns:
            The parsed Python object from YAML, or None if the extracted content is empty.

        Raises:
            ParseError: If the input cannot be parsed as a string or if the
                extracted content is invalid YAML.
        """
        # Get original text for error reporting
        yaml_string = self._fenced_parser.parse(input)
        if not yaml_string:
            return None

        try:
            return yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise ParseError(
                f'Failed to parse extracted string as YAML: {e}. String was: {yaml_string!r}',
                original_text=StrOutputParser().parse(input),
            ) from e
