"""Tests for PydanticOutputParser."""

from typing import Any
from dataclasses import dataclass

import pytest
from pydantic import Field
from pydantic import BaseModel
from pydantic import ValidationError

from tinylcel.messages import AIMessage
from tinylcel.messages import HumanMessage
from tinylcel.output_parsers import ParseError
from tinylcel.output_parsers import JsonOutputParser
from tinylcel.output_parsers.pydantic_parser import PydanticOutputParser

# --- Test Models --- #


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str
    age: int


class ComplexUser(BaseModel):
    """Complex user model with validation and optional fields."""

    username: str = Field(min_length=3, max_length=20)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    age: int = Field(ge=0, le=150)
    is_active: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class NestedModel(BaseModel):
    """Model with nested structures."""

    user: SimpleUser
    created_at: str
    settings: dict[str, Any]


class EmptyModel(BaseModel):
    """Model with no required fields."""

    optional_field: str | None = None


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


# --- Fixtures --- #


@pytest.fixture
def simple_parser() -> PydanticOutputParser[SimpleUser]:
    """Fixture for a simple PydanticOutputParser instance."""
    return PydanticOutputParser(model=SimpleUser)


@pytest.fixture
def complex_parser() -> PydanticOutputParser[ComplexUser]:
    """Fixture for a complex PydanticOutputParser instance."""
    return PydanticOutputParser(model=ComplexUser)


@pytest.fixture
def nested_parser() -> PydanticOutputParser[NestedModel]:
    """Fixture for a nested PydanticOutputParser instance."""
    return PydanticOutputParser(model=NestedModel)


@pytest.fixture
def empty_parser() -> PydanticOutputParser[EmptyModel]:
    """Fixture for an empty model PydanticOutputParser instance."""
    return PydanticOutputParser(model=EmptyModel)


# --- Successful Parsing Tests --- #


@pytest.mark.parametrize(
    ('json_content', 'expected_name', 'expected_age'),
    [
        ('{"name": "Alice", "age": 30}', 'Alice', 30),
        ('```json\n{"name": "Bob", "age": 25}\n```', 'Bob', 25),
        ('  {"name": "Charlie", "age": 35}  ', 'Charlie', 35),
        ('```\n{"name": "Diana", "age": 40}\n```', 'Diana', 40),
    ],
)
def test_simple_parser_parse_success(
    simple_parser: PydanticOutputParser[SimpleUser], json_content: str, expected_name: str, expected_age: int
) -> None:
    """Test successful parsing with simple model."""
    message = AIMessage(content=json_content)
    result = simple_parser.parse(message)

    assert isinstance(result, SimpleUser)
    assert result.name == expected_name
    assert result.age == expected_age


def test_complex_parser_parse_success(complex_parser: PydanticOutputParser[ComplexUser]) -> None:
    """Test successful parsing with complex model and validation."""
    json_content = """
    {
        "username": "johndoe",
        "email": "john@example.com",
        "age": 28,
        "is_active": true,
        "tags": ["developer", "python"],
        "metadata": {"role": "admin", "department": "engineering"}
    }
    """
    message = AIMessage(content=json_content)
    result = complex_parser.parse(message)

    assert isinstance(result, ComplexUser)
    assert result.username == 'johndoe'
    assert result.email == 'john@example.com'
    assert result.age == 28
    assert result.is_active is True
    assert result.tags == ['developer', 'python']
    assert result.metadata == {'role': 'admin', 'department': 'engineering'}


def test_nested_parser_parse_success(nested_parser: PydanticOutputParser[NestedModel]) -> None:
    """Test successful parsing with nested model."""
    json_content = """
    {
        "user": {"name": "Alice", "age": 30},
        "created_at": "2023-01-01T00:00:00Z",
        "settings": {"theme": "dark", "notifications": true}
    }
    """
    message = AIMessage(content=json_content)
    result = nested_parser.parse(message)

    assert isinstance(result, NestedModel)
    assert isinstance(result.user, SimpleUser)
    assert result.user.name == 'Alice'
    assert result.user.age == 30
    assert result.created_at == '2023-01-01T00:00:00Z'
    assert result.settings == {'theme': 'dark', 'notifications': True}


def test_empty_parser_parse_success(empty_parser: PydanticOutputParser[EmptyModel]) -> None:
    """Test successful parsing with empty model."""
    message = AIMessage(content='{}')
    result = empty_parser.parse(message)

    assert isinstance(result, EmptyModel)
    assert result.optional_field is None


def test_empty_parser_with_optional_field(empty_parser: PydanticOutputParser[EmptyModel]) -> None:
    """Test parsing with optional field provided."""
    message = AIMessage(content='{"optional_field": "test_value"}')
    result = empty_parser.parse(message)

    assert isinstance(result, EmptyModel)
    assert result.optional_field == 'test_value'


def test_parse_with_duck_typed_message(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test parsing with duck-typed message object."""
    duck_message = DuckMessage(content='{"name": "Duck", "age": 42}')
    result = simple_parser.parse(duck_message)

    assert isinstance(result, SimpleUser)
    assert result.name == 'Duck'
    assert result.age == 42


def test_parse_with_string_input(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test parsing with direct string input."""
    json_string = '{"name": "StringUser", "age": 25}'
    result = simple_parser.parse(json_string)

    assert isinstance(result, SimpleUser)
    assert result.name == 'StringUser'
    assert result.age == 25


# --- Validation Error Tests --- #


@pytest.mark.parametrize(
    ('invalid_json_content', 'expected_error_type'),
    [
        # Missing required fields
        ('{"name": "Alice"}', 'missing'),
        ('{"age": 30}', 'missing'),
        ('{}', 'missing'),
        # Wrong types
        ('{"name": 123, "age": 30}', 'string_type'),
        ('{"name": "Alice", "age": "thirty"}', 'int_parsing'),
        ('{"name": "Alice", "age": 30.5}', 'int_parsing'),
        # Extra validation for complex model
        ('{"username": "ab", "email": "valid@email.com", "age": 25}', 'string_too_short'),
        ('{"username": "valid_user", "email": "invalid-email", "age": 25}', 'string_pattern_mismatch'),
        ('{"username": "valid_user", "email": "valid@email.com", "age": -5}', 'greater_than_equal'),
        ('{"username": "valid_user", "email": "valid@email.com", "age": 200}', 'less_than_equal'),
    ],
)
def test_validation_error_cases(
    simple_parser: PydanticOutputParser[SimpleUser],
    complex_parser: PydanticOutputParser[ComplexUser],
    invalid_json_content: str,
    expected_error_type: str,
) -> None:
    """Test various Pydantic validation error cases."""
    message = AIMessage(content=invalid_json_content)

    # Choose appropriate parser based on content
    parser = complex_parser if any(field in invalid_json_content for field in ['username', 'email']) else simple_parser

    with pytest.raises(ValidationError) as exc_info:
        parser.parse(message)

    # Verify it's a Pydantic ValidationError
    assert isinstance(exc_info.value, ValidationError)

    # Check that error type matches expectation (flexible matching)
    error_details = str(exc_info.value).lower()
    if expected_error_type == 'missing':
        assert any(keyword in error_details for keyword in ['field required', 'missing', 'required'])
    elif expected_error_type in ['string_type', 'int_parsing']:
        assert any(keyword in error_details for keyword in ['type', 'parsing', 'input', 'string', 'int'])


def test_nested_validation_error(nested_parser: PydanticOutputParser[NestedModel]) -> None:
    """Test validation error in nested model."""
    json_content = """
    {
        "user": {"name": "Alice"},
        "created_at": "2023-01-01T00:00:00Z",
        "settings": {"theme": "dark"}
    }
    """
    message = AIMessage(content=json_content)

    with pytest.raises(ValidationError) as exc_info:
        nested_parser.parse(message)

    # Should fail because user.age is missing (flexible error checking)
    error_details = str(exc_info.value).lower()
    assert 'age' in error_details
    assert any(keyword in error_details for keyword in ['field required', 'missing', 'required'])


# --- JSON Parsing Error Tests --- #


@pytest.mark.parametrize(
    ('invalid_input', 'expected_exception'),
    [
        # Errors from StrOutputParser (via JsonOutputParser)
        (NotAMessage(data='test'), TypeError),
        (BadDuckMessage(content=123), ValueError),
        (123, TypeError),
        (None, TypeError),
    ],
)
def test_json_parser_error_propagation(
    simple_parser: PydanticOutputParser[SimpleUser], invalid_input: Any, expected_exception: type[BaseException]
) -> None:
    """Test that errors from the underlying JsonOutputParser are propagated."""
    with pytest.raises(expected_exception):
        simple_parser.parse(invalid_input)


@pytest.mark.parametrize(
    'bad_json_content',
    [
        '{"name": "Test", }',
        "{'name': 'Test', 'age': 30}",
        '[1, 2,]',
        'invalid json',
        '```json\n not json \n```',
    ],
)
def test_json_decode_error_propagation(simple_parser: PydanticOutputParser[SimpleUser], bad_json_content: str) -> None:
    """Test that JSONDecodeError from JsonOutputParser is wrapped in ParseError."""
    message = AIMessage(content=bad_json_content)

    with pytest.raises(ParseError) as exc_info:
        simple_parser.parse(message)

    # Check that the original text is preserved (behavior, not specific format)
    assert exc_info.value.original_text == bad_json_content


# --- Async Tests --- #


@pytest.mark.asyncio
async def test_aparse_success(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test successful async parsing."""
    json_content = '{"name": "AsyncUser", "age": 35}'
    message = AIMessage(content=json_content)

    result = await simple_parser.aparse(message)

    assert isinstance(result, SimpleUser)
    assert result.name == 'AsyncUser'
    assert result.age == 35


@pytest.mark.asyncio
async def test_aparse_validation_error(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test async parsing with validation error."""
    json_content = '{"name": "AsyncUser"}'  # Missing age
    message = AIMessage(content=json_content)

    with pytest.raises(ValidationError):
        await simple_parser.aparse(message)


@pytest.mark.asyncio
async def test_aparse_json_error(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test async parsing with JSON error."""
    bad_json_content = '{"name": invalid}'
    message = AIMessage(content=bad_json_content)

    with pytest.raises(ParseError):
        await simple_parser.aparse(message)


# --- Invoke/AInvoke Tests --- #


def test_invoke_success(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test successful sync invocation."""
    message = AIMessage(content='{"name": "InvokeUser", "age": 40}')
    result = simple_parser.invoke(message)

    assert isinstance(result, SimpleUser)
    assert result.name == 'InvokeUser'
    assert result.age == 40


def test_invoke_validation_error(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test sync invocation with validation error."""
    message = AIMessage(content='{"name": 123, "age": 40}')  # Invalid name type

    with pytest.raises(ValidationError):
        simple_parser.invoke(message)


def test_invoke_json_error(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test sync invocation with JSON error."""
    message = AIMessage(content='not json')

    with pytest.raises(ParseError):
        simple_parser.invoke(message)


@pytest.mark.asyncio
async def test_ainvoke_success(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test successful async invocation."""
    message = AIMessage(content='{"name": "AInvokeUser", "age": 45}')
    result = await simple_parser.ainvoke(message)

    assert isinstance(result, SimpleUser)
    assert result.name == 'AInvokeUser'
    assert result.age == 45


@pytest.mark.asyncio
async def test_ainvoke_validation_error(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test async invocation with validation error."""
    message = AIMessage(content='{"age": 45}')  # Missing name

    with pytest.raises(ValidationError):
        await simple_parser.ainvoke(message)


@pytest.mark.asyncio
async def test_ainvoke_json_error(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test async invocation with JSON error."""
    message = AIMessage(content='{key: value}')

    with pytest.raises(ParseError):
        await simple_parser.ainvoke(message)


# --- Parser Configuration and Behavior Tests --- #


def test_parser_with_custom_json_parser_behavior() -> None:
    """Test that parser works correctly with custom JsonOutputParser configuration."""
    custom_json_parser = JsonOutputParser()
    parser = PydanticOutputParser(model=SimpleUser, parser=custom_json_parser)

    message = AIMessage(content='{"name": "CustomParser", "age": 50}')
    result = parser.parse(message)

    # Focus on behavior: parser should work correctly regardless of internal configuration
    assert isinstance(result, SimpleUser)
    assert result.name == 'CustomParser'
    assert result.age == 50


def test_parser_configuration_independence() -> None:
    """Test that different parser instances are independent."""
    parser1 = PydanticOutputParser(model=SimpleUser)
    parser2 = PydanticOutputParser(model=SimpleUser)

    message = AIMessage(content='{"name": "Test", "age": 30}')

    result1 = parser1.parse(message)
    result2 = parser2.parse(message)

    # Both should work independently and produce equivalent results
    assert isinstance(result1, SimpleUser)
    assert isinstance(result2, SimpleUser)
    assert result1.name == result2.name == 'Test'
    assert result1.age == result2.age == 30


def test_parser_state_consistency() -> None:
    """Test that parser maintains consistent behavior across multiple calls."""
    parser = PydanticOutputParser(model=SimpleUser)
    message = AIMessage(content='{"name": "Consistent", "age": 25}')

    # Multiple calls should produce equivalent results
    result1 = parser.parse(message)
    result2 = parser.parse(message)

    assert isinstance(result1, SimpleUser)
    assert isinstance(result2, SimpleUser)
    assert result1.name == result2.name == 'Consistent'
    assert result1.age == result2.age == 25


# --- Edge Cases and Integration Tests --- #


def test_parse_with_extra_fields(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test parsing JSON with extra fields (should be ignored by Pydantic)."""
    json_content = """
    {
        "name": "ExtraFields",
        "age": 30,
        "extra_field": "ignored",
        "another_extra": 123
    }
    """
    message = AIMessage(content=json_content)
    result = simple_parser.parse(message)

    assert isinstance(result, SimpleUser)
    assert result.name == 'ExtraFields'
    assert result.age == 30
    # Extra fields should be ignored
    assert not hasattr(result, 'extra_field')
    assert not hasattr(result, 'another_extra')


def test_parse_with_null_values(empty_parser: PydanticOutputParser[EmptyModel]) -> None:
    """Test parsing JSON with null values."""
    json_content = '{"optional_field": null}'
    message = AIMessage(content=json_content)
    result = empty_parser.parse(message)

    assert isinstance(result, EmptyModel)
    assert result.optional_field is None


def test_parse_complex_nested_structure() -> None:
    """Test parsing complex nested JSON structure."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Company(BaseModel):
        name: str
        address: Address

    class Employee(BaseModel):
        name: str
        age: int
        company: Company
        skills: list[str]

    parser = PydanticOutputParser(model=Employee)

    json_content = """
    {
        "name": "John Doe",
        "age": 30,
        "company": {
            "name": "Tech Corp",
            "address": {
                "street": "123 Main St",
                "city": "San Francisco",
                "country": "USA"
            }
        },
        "skills": ["Python", "Machine Learning", "Docker"]
    }
    """

    message = AIMessage(content=json_content)
    result = parser.parse(message)

    assert isinstance(result, Employee)
    assert result.name == 'John Doe'
    assert result.age == 30
    assert isinstance(result.company, Company)
    assert result.company.name == 'Tech Corp'
    assert isinstance(result.company.address, Address)
    assert result.company.address.street == '123 Main St'
    assert result.company.address.city == 'San Francisco'
    assert result.company.address.country == 'USA'
    assert result.skills == ['Python', 'Machine Learning', 'Docker']


def test_parser_interface_contract(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test that parser implements the expected interface contract."""
    # Test that required methods exist and are callable
    assert hasattr(simple_parser, 'parse')
    assert callable(simple_parser.parse)
    assert hasattr(simple_parser, 'aparse')
    assert callable(simple_parser.aparse)
    assert hasattr(simple_parser, 'invoke')
    assert callable(simple_parser.invoke)
    assert hasattr(simple_parser, 'ainvoke')
    assert callable(simple_parser.ainvoke)


def test_parser_with_different_message_types(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test parser works with different message types."""
    json_content = '{"name": "MessageTest", "age": 25}'

    # Test with different message types
    ai_message = AIMessage(content=json_content)
    human_message = HumanMessage(content=json_content)

    ai_result = simple_parser.parse(ai_message)
    human_result = simple_parser.parse(human_message)

    assert isinstance(ai_result, SimpleUser)
    assert isinstance(human_result, SimpleUser)
    assert ai_result.name == human_result.name == 'MessageTest'
    assert ai_result.age == human_result.age == 25


@pytest.mark.asyncio
async def test_async_sync_behavior_equivalence(simple_parser: PydanticOutputParser[SimpleUser]) -> None:
    """Test that async and sync methods produce equivalent results."""
    message = AIMessage(content='{"name": "EquivalenceTest", "age": 35}')

    sync_result = simple_parser.parse(message)
    async_result = await simple_parser.aparse(message)

    # Results should be equivalent
    assert isinstance(sync_result, type(async_result))
    assert sync_result.name == async_result.name
    assert sync_result.age == async_result.age

    # Test invoke methods too
    sync_invoke_result = simple_parser.invoke(message)
    async_invoke_result = await simple_parser.ainvoke(message)

    assert isinstance(sync_invoke_result, type(async_invoke_result))
    assert sync_invoke_result.name == async_invoke_result.name
    assert sync_invoke_result.age == async_invoke_result.age
