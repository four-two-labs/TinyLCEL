"""Output parsers for TinyLCEL."""

import re
import abc
import json
from typing import Any
from typing import Protocol
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

    # Inherited invoke/ainvoke will call parse/aparse
    def invoke(self, input: Any) -> OutputType:
        """Parse the input synchronously."""
        return self.parse(input)

    async def ainvoke(self, input: Any) -> OutputType:
        """Parse the input asynchronously."""
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


@dataclass(frozen=True)
class JsonOutputParser(StrOutputParser, RunnableBase[str, dict]):
    """
    Parses JSON output from an LLM call.

    Handles potential markdown code fences (e.g., ```json ... ```) and
    leading/trailing whitespace.
    Expects the input to the parser to have a string 'content' attribute
    (like AIMessage).
    """

    _json_regex: re.Pattern = re.compile(r'^\s*(?:```(?:json)?\n*)?(.*?)(?:\n*```)?\s*$', re.IGNORECASE | re.DOTALL)

    def parse(self, input: Any) -> Any:
        """
        Parses the string input into a Python object via JSON.

        Args:
            input: An object assumed to have a 'content' attribute (e.g., AIMessage).

        Returns:
            The parsed Python object (dict, list, str, int, float, bool, None).

        Raises:
            ParseEx: If the input cannot be parsed as a string,
                if the regex fails to find a suitable JSON block, or if the
                extracted block is invalid JSON.
        """
        text = super().parse(input)
        match = self._json_regex.search(text)
        if match is None:
            raise ParseError(f'Could not extract JSON block from received text: {text}', original_text=text)

        json_string = match.group(1).strip()
        if not json_string:
            return {}

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ParseError(
                f'Failed to parse extracted string as JSON: {e}. String was: {json_string!r}', original_text=text
            ) from e


@dataclass(frozen=True)
class YamlOutputParser(StrOutputParser):
    """
    Parses YAML output from an LLM call.

    Handles potential markdown code fences (e.g., ```yaml ... ```) and
    leading/trailing whitespace using a regex.
    Expects the input to the parser to have a string 'content' attribute
    (like AIMessage).
    Uses yaml.safe_load to prevent arbitrary code execution.
    """

    # Regex to extract content within optional yaml/markdown fences
    _yaml_regex: re.Pattern = re.compile(
        # Same regex as JSON, language tag is optional and case-insensitive
        r'^\s*(?:```(?:yaml)?\n*)?(.*?)(?:\n*```)?\s*$',
        re.IGNORECASE | re.DOTALL,
    )

    def parse(self, input: Any) -> Any:
        """
        Parses the string input as YAML after extracting via regex.

        Args:
            input: An object assumed to have a 'content' attribute (e.g., AIMessage).

        Returns:
            The parsed Python object from YAML, or None if the extracted content is empty.

        Raises:
            ParseException: If the input cannot be parsed as a string, if the regex
                fails to find a suitable YAML block, or if the extracted block
                is invalid YAML.
        """
        text = super().parse(input)
        match = self._yaml_regex.search(text)
        if match is None:
            raise ParseError(f'Could not extract YAML block from received text: {text}', original_text=text)

        yaml_string = match.group(1).strip()
        if not yaml_string:
            return None

        try:
            return yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise ParseError(
                f'Failed to parse extracted string as YAML: {e}. String was: {yaml_string!r}', original_text=text
            ) from e
