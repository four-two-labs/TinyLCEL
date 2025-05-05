"""Tests for output parsers."""

import pytest
from dataclasses import dataclass
from typing import Any

from tinylcel.messages import AIMessage
from tinylcel.messages import HumanMessage
from tinylcel.output_parsers import StrOutputParser
from tinylcel.output_parsers import JsonOutputParser
from tinylcel.output_parsers import ParseException


# --- Fixtures --- S

@pytest.fixture
def parser() -> StrOutputParser:
    """Fixture for a StrOutputParser instance."""
    return StrOutputParser()

@pytest.fixture
def json_parser() -> JsonOutputParser:
    """Fixture for a JsonOutputParser instance."""
    return JsonOutputParser()

@dataclass
class DuckMessage:
    """A class that isn't a BaseMessage but has a content attribute."""
    content: str

@dataclass
class BadDuckMessage:
    """A class that has a non-string content attribute."""
    content: int

@dataclass
class NotAMessage:
    """A class without a content attribute."""
    data: str


# --- Test Cases --- #

# StrOutputParser.parse tests

@pytest.mark.parametrize(
    "input_message, expected_output",
    [
        (AIMessage(content='Hello AI'), 'Hello AI'),
        (HumanMessage(content='Hello Human'), 'Hello Human'), # Any BaseMessage
        (DuckMessage(content='Hello Duck'), 'Hello Duck'), # Duck typing
        (AIMessage(content=''), ''), # Empty string
    ]
)
def test_str_output_parser_parse_success(parser: StrOutputParser, input_message, expected_output: str):
    """Test successful parsing of various message types."""
    assert parser.parse(input_message) == expected_output

@pytest.mark.parametrize(
    "invalid_input, expected_exception, match_message",
    [
        (NotAMessage(data='test'), TypeError, r"^Expected object with 'content' attribute, got .*NotAMessage"),
        (BadDuckMessage(content=123), ValueError, r"^Expected string content, got <class 'int'>"),
        (123, TypeError, r"^Expected object with 'content' attribute, got <class 'int'>"),
        (None, TypeError, r"^Expected object with 'content' attribute, got <class 'NoneType'>"),
        (['list'], TypeError, r"^Expected object with 'content' attribute, got <class 'list'>"),
    ]
)
def test_str_output_parser_parse_failure(parser: StrOutputParser, invalid_input, expected_exception, match_message: str):
    """Test parsing failures with various invalid inputs."""
    with pytest.raises(expected_exception, match=match_message):
        parser.parse(invalid_input)

# StrOutputParser.aparse tests (uses parse via asyncio.to_thread)

@pytest.mark.asyncio
async def test_str_output_parser_aparse_success(parser: StrOutputParser):
    """Test successful async parsing."""
    message = AIMessage(content='Async Hello')
    assert await parser.aparse(message) == 'Async Hello'

@pytest.mark.asyncio
async def test_str_output_parser_aparse_failure_type(parser: StrOutputParser):
    """Test async parsing type error propagation."""
    invalid_input = NotAMessage(data='test')
    with pytest.raises(TypeError, match=r"^Expected object with 'content' attribute, got .*NotAMessage"):
        await parser.aparse(invalid_input)

@pytest.mark.asyncio
async def test_str_output_parser_aparse_failure_value(parser: StrOutputParser):
    """Test async parsing value error propagation."""
    invalid_input = BadDuckMessage(content=123)
    with pytest.raises(ValueError, match=r"^Expected string content, got <class 'int'>"):
        await parser.aparse(invalid_input)

# StrOutputParser.invoke tests (calls parse)

def test_str_output_parser_invoke_success(parser: StrOutputParser):
    """Test successful sync invocation."""
    message = AIMessage(content='Invoke Hello')
    assert parser.invoke(message) == 'Invoke Hello'

def test_str_output_parser_invoke_failure_type(parser: StrOutputParser):
    """Test sync invocation type error propagation."""
    invalid_input = NotAMessage(data='test')
    with pytest.raises(TypeError, match=r"^Expected object with 'content' attribute, got .*NotAMessage"):
        parser.invoke(invalid_input)

def test_str_output_parser_invoke_failure_value(parser: StrOutputParser):
    """Test sync invocation value error propagation."""
    invalid_input = BadDuckMessage(content=123)
    with pytest.raises(ValueError, match=r"^Expected string content, got <class 'int'>"):
        parser.invoke(invalid_input)

# StrOutputParser.ainvoke tests (calls aparse)

@pytest.mark.asyncio
async def test_str_output_parser_ainvoke_success(parser: StrOutputParser):
    """Test successful async invocation."""
    message = AIMessage(content='AInvoke Hello')
    assert await parser.ainvoke(message) == 'AInvoke Hello'

@pytest.mark.asyncio
async def test_str_output_parser_ainvoke_failure_type(parser: StrOutputParser):
    """Test async invocation type error propagation."""
    invalid_input = NotAMessage(data='test')
    with pytest.raises(TypeError, match=r"^Expected object with 'content' attribute, got .*NotAMessage"):
        await parser.ainvoke(invalid_input)

@pytest.mark.asyncio
async def test_str_output_parser_ainvoke_failure_value(parser: StrOutputParser):
    """Test async invocation value error propagation."""
    invalid_input = BadDuckMessage(content=123)
    with pytest.raises(ValueError, match=r"^Expected string content, got <class 'int'>"):
        await parser.ainvoke(invalid_input)

# --- JsonOutputParser Tests --- #

@pytest.mark.parametrize(
    "raw_content, expected_parsed",
    [
        # Simple cases
        ('{"name": "Test", "value": 123}', {"name": "Test", "value": 123}),
        ('[1, "two", null, true]', [1, "two", None, True]),
        # With markdown fences (json)
        ('```json\n{"key": "value"}\n```', {"key": "value"}),
        ('```JSON\n[1, 2]\n```', [1, 2]), # Case insensitive fence
        # With markdown fences (no lang tag)
        ('```\n{"key": "value"}\n```', {"key": "value"}),
        # With surrounding whitespace/newlines
        ('\n   {\n"a": 1\n}  \n', {"a": 1}),
        (' \n ```json\n ["test"] \n``` \n ', ["test"]),
        # Empty content (after stripping fences/whitespace)
        ('', {}),
        ('```json\n```', {}),
        ('  \n ', {}),
        # Empty but valid JSON structures
        ('{}', {}),
        ('[]', []),
        # Missing closing fence
        ('```json\n{"unclosed": true}', {"unclosed": True}),
        ('```\n{"unclosed_no_lang": true}', {"unclosed_no_lang": True}),
        ('  {\n"no_fence_at_all": false\n} \n', {"no_fence_at_all": False}), # No fences at all
        # Multi-line JSON
        ('{\n  "key": "value",\n  "another": [\n    1,\n    2\n  ]\n}', {"key": "value", "another": [1, 2]}),
        ('```json\n{\n  "multi": true,\n  "line": "yes"\n}\n```', {"multi": True, "line": "yes"}),
    ]
)
def test_json_output_parser_parse_success(json_parser: JsonOutputParser, raw_content: str, expected_parsed: Any):
    """Test successful JSON parsing with various formats."""
    message = AIMessage(content=raw_content)
    assert json_parser.parse(message) == expected_parsed

@pytest.mark.parametrize(
    "invalid_input, expected_exception, match_message",
    [
        # These should fail in the parent StrOutputParser first
        (NotAMessage(data='test'), TypeError, r"^Expected object with 'content' attribute"),
        (123, TypeError, r"^Expected object with 'content' attribute"),
        (None, TypeError, r"^Expected object with 'content' attribute"),
        # This has content, but it's not a string
        (BadDuckMessage(content=123), ValueError, r"^Expected string content, got <class 'int'>"),
    ]
)
def test_json_output_parser_parse_failure_str_parser(json_parser: JsonOutputParser, invalid_input: Any, expected_exception: type[Exception], match_message: str):
    """Test that errors from the parent StrOutputParser are propagated."""
    with pytest.raises(expected_exception, match=match_message):
        json_parser.parse(invalid_input)

@pytest.mark.parametrize(
    "bad_json_content, match_message",
    [
        ('{"name": "Test", }', r"^Failed to parse extracted string as JSON: Illegal trailing comma"),
        ("{'key': 'value'}", r"^Failed to parse extracted string as JSON: Expecting property name enclosed in double quotes"),
        ('[1, 2,]', r"^Failed to parse extracted string as JSON: Illegal trailing comma"),
        ('invalid json', r"^Failed to parse extracted string as JSON: Expecting value"),
        ('```json\n not json \n```', r"^Failed to parse extracted string as JSON: Expecting value"),
    ]
)
def test_json_output_parser_parse_failure_json_decode(json_parser: JsonOutputParser, bad_json_content: str, match_message: str):
    """Test that JSONDecodeError is caught and wrapped in ParseException."""
    message = AIMessage(content=bad_json_content)
    with pytest.raises(ParseException, match=match_message) as exc_info:
        json_parser.parse(message)
    # Check that the original text is stored
    assert exc_info.value.original_text == bad_json_content

# --- Async Tests for JsonOutputParser --- #

@pytest.mark.asyncio
async def test_json_output_parser_aparse_success(json_parser: JsonOutputParser):
    """Test successful async JSON parsing."""
    raw_content = '```json\n{"result": true}\n```'
    message = AIMessage(content=raw_content)
    assert await json_parser.aparse(message) == {"result": True}

@pytest.mark.asyncio
async def test_json_output_parser_aparse_failure_json_decode(json_parser: JsonOutputParser):
    """Test async JSONDecodeError failure."""
    bad_json_content = '{"key": invalid}'
    message = AIMessage(content=bad_json_content)
    with pytest.raises(ParseException, match=r"^Failed to parse extracted string as JSON") as exc_info:
        await json_parser.aparse(message)
    assert exc_info.value.original_text == bad_json_content

# --- Invoke/AInvoke Tests for JsonOutputParser --- #

def test_json_output_parser_invoke_success(json_parser: JsonOutputParser):
    """Test successful sync invocation."""
    message = AIMessage(content='{"data": [1, 2]}')
    assert json_parser.invoke(message) == {"data": [1, 2]}

def test_json_output_parser_invoke_failure_json_decode(json_parser: JsonOutputParser):
    """Test sync invocation JSONDecodeError failure."""
    message = AIMessage(content='not json')
    with pytest.raises(ParseException, match=r"^Failed to parse extracted string as JSON"):
        json_parser.invoke(message)

@pytest.mark.asyncio
async def test_json_output_parser_ainvoke_success(json_parser: JsonOutputParser):
    """Test successful async invocation."""
    message = AIMessage(content=' ["a", "b"] ')
    assert await json_parser.ainvoke(message) == ["a", "b"]

@pytest.mark.asyncio
async def test_json_output_parser_ainvoke_failure_json_decode(json_parser: JsonOutputParser):
    """Test async invocation JSONDecodeError failure."""
    message = AIMessage(content='{key: value}')
    with pytest.raises(ParseException, match=r"^Failed to parse extracted string as JSON"):
        await json_parser.ainvoke(message) 