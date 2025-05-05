"""Tests for output parsers."""

import pytest
from dataclasses import dataclass

from tinylcel.messages import AIMessage
from tinylcel.messages import HumanMessage
from tinylcel.output_parsers import StrOutputParser


# --- Fixtures --- S

@pytest.fixture
def parser() -> StrOutputParser:
    """Fixture for a StrOutputParser instance."""
    return StrOutputParser()

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