"""Tests for output parsers."""

from typing import Any
from typing import Literal
from dataclasses import dataclass

import pytest

from tinylcel.messages import AIMessage
from tinylcel.messages import HumanMessage
from tinylcel.output_parsers import ParseError
from tinylcel.output_parsers import StrOutputParser
from tinylcel.output_parsers import JsonOutputParser
from tinylcel.output_parsers import YamlOutputParser
from tinylcel.output_parsers import FenceOutputParser
from tinylcel.output_parsers.base_parsers import BaseOutputParser

# --- Fixtures --- S


@pytest.fixture
def parser() -> StrOutputParser:
    """Fixture for a StrOutputParser instance."""
    return StrOutputParser()


@pytest.fixture
def json_parser() -> JsonOutputParser:
    """Fixture for a JsonOutputParser instance."""
    return JsonOutputParser()


@pytest.fixture
def yaml_parser() -> YamlOutputParser:
    """Fixture for a YamlOutputParser instance."""
    return YamlOutputParser()


@pytest.fixture
def python_fenced_parser() -> BaseOutputParser[str]:
    """Fixture for a Python-specific FenceOutputParser instance."""
    return FenceOutputParser('python')


@pytest.fixture
def generic_fenced_parser() -> BaseOutputParser[str]:
    """Fixture for a generic FenceOutputParser instance."""
    return FenceOutputParser()


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
    ('input_message', 'expected_output'),
    [
        (AIMessage(content='Hello AI'), 'Hello AI'),
        (HumanMessage(content='Hello Human'), 'Hello Human'),  # Any BaseMessage
        (DuckMessage(content='Hello Duck'), 'Hello Duck'),  # Duck typing
        (AIMessage(content=''), ''),  # Empty string
    ],
)
def test_str_output_parser_parse_success(parser: StrOutputParser, input_message: Any, expected_output: str) -> None:
    """Test successful parsing of various message types."""
    assert parser.parse(input_message) == expected_output


@pytest.mark.parametrize(
    ('invalid_input', 'expected_exception'),
    [
        (NotAMessage(data='test'), TypeError),
        (BadDuckMessage(content=123), ValueError),
        (123, TypeError),
        (None, TypeError),
        (['list'], TypeError),
    ],
)
def test_str_output_parser_parse_failure(
    parser: StrOutputParser, invalid_input: Any, expected_exception: type[BaseException]
) -> None:
    """Test parsing failures with various invalid inputs."""
    with pytest.raises(expected_exception):
        parser.parse(invalid_input)


# StrOutputParser.aparse tests (uses parse via asyncio.to_thread)


@pytest.mark.asyncio
async def test_str_output_parser_aparse_success(parser: StrOutputParser) -> None:
    """Test successful async parsing."""
    message = AIMessage(content='Async Hello')
    assert await parser.aparse(message) == 'Async Hello'


@pytest.mark.asyncio
async def test_str_output_parser_aparse_failure_type(parser: StrOutputParser) -> None:
    """Test async parsing type error propagation."""
    invalid_input = NotAMessage(data='test')
    with pytest.raises(TypeError):
        await parser.aparse(invalid_input)


@pytest.mark.asyncio
async def test_str_output_parser_aparse_failure_value(parser: StrOutputParser) -> None:
    """Test async parsing value error propagation."""
    invalid_input = BadDuckMessage(content=123)
    with pytest.raises(ValueError, match='Expected string content'):
        await parser.aparse(invalid_input)


# StrOutputParser.invoke tests (calls parse)


def test_str_output_parser_invoke_success(parser: StrOutputParser) -> None:
    """Test successful sync invocation."""
    message = AIMessage(content='Invoke Hello')
    assert parser.invoke(message) == 'Invoke Hello'


def test_str_output_parser_invoke_failure_type(parser: StrOutputParser) -> None:
    """Test sync invocation type error propagation."""
    invalid_input = NotAMessage(data='test')
    with pytest.raises(TypeError):
        parser.invoke(invalid_input)


def test_str_output_parser_invoke_failure_value(parser: StrOutputParser) -> None:
    """Test sync invocation value error propagation."""
    invalid_input = BadDuckMessage(content=123)
    with pytest.raises(ValueError, match='Expected string content'):
        parser.invoke(invalid_input)


# StrOutputParser.ainvoke tests (calls aparse)


@pytest.mark.asyncio
async def test_str_output_parser_ainvoke_success(parser: StrOutputParser) -> None:
    """Test successful async invocation."""
    message = AIMessage(content='AInvoke Hello')
    assert await parser.ainvoke(message) == 'AInvoke Hello'


@pytest.mark.asyncio
async def test_str_output_parser_ainvoke_failure_type(parser: StrOutputParser) -> None:
    """Test async invocation type error propagation."""
    invalid_input = NotAMessage(data='test')
    with pytest.raises(TypeError):
        await parser.ainvoke(invalid_input)


@pytest.mark.asyncio
async def test_str_output_parser_ainvoke_failure_value(parser: StrOutputParser) -> None:
    """Test async invocation value error propagation."""
    invalid_input = BadDuckMessage(content=123)
    with pytest.raises(ValueError, match='Expected string content'):
        await parser.ainvoke(invalid_input)


# --- JsonOutputParser Tests --- #


@pytest.mark.parametrize(
    ('raw_content', 'expected_parsed'),
    [
        # Simple cases
        ('{"name": "Test", "value": 123}', {'name': 'Test', 'value': 123}),
        ('[1, "two", null, true]', [1, 'two', None, True]),
        # With markdown fences (json)
        ('```json\n{"key": "value"}\n```', {'key': 'value'}),
        ('```JSON\n[1, 2]\n```', [1, 2]),  # Case insensitive fence
        # With markdown fences (no lang tag)
        ('```\n{"key": "value"}\n```', {'key': 'value'}),
        # With surrounding whitespace/newlines
        ('\n   {\n"a": 1\n}  \n', {'a': 1}),
        (' \n ```json\n ["test"] \n``` \n ', ['test']),
        # Empty content (after stripping fences/whitespace)
        ('', {}),
        ('```json\n```', {}),
        ('  \n ', {}),
        # Empty but valid JSON structures
        ('{}', {}),
        ('[]', []),
        # Missing closing fence
        ('```json\n{"unclosed": true}', {'unclosed': True}),
        ('```\n{"unclosed_no_lang": true}', {'unclosed_no_lang': True}),
        ('  {\n"no_fence_at_all": false\n} \n', {'no_fence_at_all': False}),  # No fences at all
        # Multi-line JSON
        ('{\n  "key": "value",\n  "another": [\n    1,\n    2\n  ]\n}', {'key': 'value', 'another': [1, 2]}),
        ('```json\n{\n  "multi": true,\n  "line": "yes"\n}\n```', {'multi': True, 'line': 'yes'}),
    ],
)
def test_json_output_parser_parse_success(
    json_parser: JsonOutputParser, raw_content: str, expected_parsed: Any
) -> None:
    """Test successful JSON parsing with various formats."""
    message = AIMessage(content=raw_content)
    assert json_parser.parse(message) == expected_parsed


@pytest.mark.parametrize(
    ('invalid_input', 'expected_exception'),
    [
        # These should fail in the parent StrOutputParser first
        (NotAMessage(data='test'), TypeError),
        (123, TypeError),
        (None, TypeError),
        # This has content, but it's not a string
        (BadDuckMessage(content=123), ValueError),
    ],
)
def test_json_output_parser_parse_failure_str_parser(
    json_parser: JsonOutputParser, invalid_input: Any, expected_exception: type[BaseException]
) -> None:
    """Test that errors from the parent StrOutputParser are propagated."""
    with pytest.raises(expected_exception):
        json_parser.parse(invalid_input)


@pytest.mark.parametrize(
    'bad_json_content',
    [
        '{"name": "Test", }',
        "{'key': 'value'}",
        '[1, 2,]',
        'invalid json',
        '```json\n not json \n```',
    ],
)
def test_json_output_parser_parse_failure_json_decode(json_parser: JsonOutputParser, bad_json_content: str) -> None:
    """Test that JSONDecodeError is caught and wrapped in ParseError."""
    message = AIMessage(content=bad_json_content)
    with pytest.raises(ParseError) as exc_info:
        json_parser.parse(message)
    # Check that the original text is stored (behavior, not specific format)
    assert exc_info.value.original_text == bad_json_content


# --- Async Tests for JsonOutputParser --- #


@pytest.mark.asyncio
async def test_json_output_parser_aparse_success(json_parser: JsonOutputParser) -> None:
    """Test successful async JSON parsing."""
    raw_content = '```json\n{"result": true}\n```'
    message = AIMessage(content=raw_content)
    assert await json_parser.aparse(message) == {'result': True}


@pytest.mark.asyncio
async def test_json_output_parser_aparse_failure_json_decode(json_parser: JsonOutputParser) -> None:
    """Test async JSONDecodeError failure."""
    bad_json_content = '{"key": invalid}'
    message = AIMessage(content=bad_json_content)
    with pytest.raises(ParseError) as exc_info:
        await json_parser.aparse(message)
    assert exc_info.value.original_text == bad_json_content


# --- Invoke/AInvoke Tests for JsonOutputParser --- #


def test_json_output_parser_invoke_success(json_parser: JsonOutputParser) -> None:
    """Test successful sync invocation."""
    message = AIMessage(content='{"data": [1, 2]}')
    assert json_parser.invoke(message) == {'data': [1, 2]}


def test_json_output_parser_invoke_failure_json_decode(json_parser: JsonOutputParser) -> None:
    """Test sync invocation JSONDecodeError failure."""
    message = AIMessage(content='not json')
    with pytest.raises(ParseError):
        json_parser.invoke(message)


@pytest.mark.asyncio
async def test_json_output_parser_ainvoke_success(json_parser: JsonOutputParser) -> None:
    """Test successful async invocation."""
    message = AIMessage(content=' ["a", "b"] ')
    assert await json_parser.ainvoke(message) == ['a', 'b']


@pytest.mark.asyncio
async def test_json_output_parser_ainvoke_failure_json_decode(json_parser: JsonOutputParser) -> None:
    """Test async invocation JSONDecodeError failure."""
    message = AIMessage(content='{key: value}')
    with pytest.raises(ParseError):
        await json_parser.ainvoke(message)


# --- YamlOutputParser Tests (Regex version) --- #


@pytest.mark.parametrize(
    ('raw_content', 'expected_parsed'),
    [
        # Simple cases
        ('name: Test\nvalue: 123', {'name': 'Test', 'value': 123}),
        ('- 1\n- two\n- null\n- true', [1, 'two', None, True]),
        # With markdown fences (yaml)
        ('```yaml\nkey: value\n```', {'key': 'value'}),
        ('```YAML\n- 1\n- 2\n```', [1, 2]),  # Case insensitive fence
        # With markdown fences (no lang tag)
        ('```\nkey: value\n```', {'key': 'value'}),
        # With surrounding whitespace/newlines
        ('\n   \nkey: value\n  \n', {'key': 'value'}),
        (' \n ```yaml\n - test \n``` \n ', ['test']),
        # Empty content (after stripping fences/whitespace via regex)
        ('', None),  # Regex group(1) will be empty -> None
        ('```yaml\n```', None),
        ('  \n ', None),
        # Empty but valid YAML structures
        ('{}', {}),
        ('[]', []),
        # Multi-line
        (
            'parent:\n  child: 1\n  list:\n    - item1\n    - item2',
            {'parent': {'child': 1, 'list': ['item1', 'item2']}},
        ),
        # Missing closing fence
        ('```yaml\nunclosed: true', {'unclosed': True}),
        ('```\n  no_closing_no_lang: yes', {'no_closing_no_lang': True}),
    ],
)
def test_yaml_output_parser_parse_success(
    yaml_parser: YamlOutputParser, raw_content: str, expected_parsed: Any
) -> None:
    """Test successful YAML parsing with various formats using regex extraction."""
    message = AIMessage(content=raw_content)
    assert yaml_parser.parse(message) == expected_parsed


@pytest.mark.parametrize(
    ('invalid_input', 'expected_exception'),
    [
        # Errors from parent StrOutputParser
        (NotAMessage(data='test'), TypeError),
        (BadDuckMessage(content=123), ValueError),
        (123, TypeError),
        (None, TypeError),
    ],
)
def test_yaml_output_parser_parse_failure_str_parser(
    yaml_parser: YamlOutputParser, invalid_input: Any, expected_exception: type[BaseException]
) -> None:
    """Test that errors from the parent StrOutputParser are propagated for YAML."""
    with pytest.raises(expected_exception):
        yaml_parser.parse(invalid_input)


@pytest.mark.parametrize(
    'bad_yaml_content',
    [
        'unbalanced: {key: value',
        '```yaml\n invalid : yaml : here \n```',
        'key: value\n  bad_indent: true',
    ],
)
def test_yaml_output_parser_parse_failure_yaml_error(yaml_parser: YamlOutputParser, bad_yaml_content: str) -> None:
    """Test that YAMLError is caught and wrapped in ParseError."""
    message = AIMessage(content=bad_yaml_content)
    with pytest.raises(ParseError) as exc_info:
        yaml_parser.parse(message)
    assert exc_info.value.original_text == bad_yaml_content


# --- Async Tests for YamlOutputParser --- #


@pytest.mark.asyncio
async def test_yaml_output_parser_aparse_success(yaml_parser: YamlOutputParser) -> None:
    """Test successful async YAML parsing."""
    raw_content = '```yaml\nresult: true\n```'
    message = AIMessage(content=raw_content)
    assert await yaml_parser.aparse(message) == {'result': True}


@pytest.mark.asyncio
async def test_yaml_output_parser_aparse_failure_yaml_error(yaml_parser: YamlOutputParser) -> None:
    """Test async YAMLError failure."""
    bad_yaml_content = 'key: {invalid: yaml\n'  # Unbalanced braces
    message = AIMessage(content=bad_yaml_content)
    # Focus on behavior: should raise ParseError, not specific message format
    with pytest.raises(ParseError) as exc_info:
        await yaml_parser.aparse(message)
    assert exc_info.value.original_text == bad_yaml_content


# --- Invoke/AInvoke Tests for YamlOutputParser --- #


def test_yaml_output_parser_invoke_success(yaml_parser: YamlOutputParser) -> None:
    """Test successful sync YAML invocation."""
    message = AIMessage(content='- item1\n- item2')
    assert yaml_parser.invoke(message) == ['item1', 'item2']


def test_yaml_output_parser_invoke_failure_yaml_error(yaml_parser: YamlOutputParser) -> None:
    """Test sync invocation YAMLError failure."""
    message = AIMessage(content='key : bad : yaml')
    with pytest.raises(ParseError):
        yaml_parser.invoke(message)


@pytest.mark.asyncio
async def test_yaml_output_parser_ainvoke_success(yaml_parser: YamlOutputParser) -> None:
    """Test successful async YAML invocation."""
    message = AIMessage(content='field: value')
    assert await yaml_parser.ainvoke(message) == {'field': 'value'}


@pytest.mark.asyncio
async def test_yaml_output_parser_ainvoke_failure_yaml_error(yaml_parser: YamlOutputParser) -> None:
    """Test async invocation YAMLError failure."""
    message = AIMessage(content='key: [unclosed')
    with pytest.raises(ParseError):
        await yaml_parser.ainvoke(message)


# --- Parser Behavior and Contract Tests --- #


@pytest.mark.parametrize(
    'parser_instance',
    [
        StrOutputParser(),
        JsonOutputParser(),
        YamlOutputParser(),
        FenceOutputParser('python'),
    ],
)
def test_parser_interface_contract(parser_instance: BaseOutputParser[Any]) -> None:
    """Test that all parsers implement the expected interface."""
    assert hasattr(parser_instance, 'parse')
    assert callable(parser_instance.parse)
    assert hasattr(parser_instance, 'aparse')
    assert callable(parser_instance.aparse)
    assert hasattr(parser_instance, 'invoke')
    assert callable(parser_instance.invoke)
    assert hasattr(parser_instance, 'ainvoke')
    assert callable(parser_instance.ainvoke)


@pytest.mark.parametrize(
    ('parser_instance', 'test_message', 'expected_result'),
    [
        (StrOutputParser(), AIMessage(content='test string'), 'test string'),
        (JsonOutputParser(), AIMessage(content='{"key": "value"}'), {'key': 'value'}),
        (YamlOutputParser(), AIMessage(content='key: value'), {'key': 'value'}),
        (FenceOutputParser('python'), AIMessage(content='```python\ntest_code = True\n```'), 'test_code = True'),
        (FenceOutputParser(), AIMessage(content='```js\nalert("hello")\n```'), 'alert("hello")'),
    ],
)
@pytest.mark.asyncio
async def test_parser_async_sync_equivalence(
    parser_instance: BaseOutputParser[Any], test_message: AIMessage, expected_result: Any
) -> None:
    """Test that async and sync methods produce equivalent results across parsers."""
    sync_result = parser_instance.parse(test_message)
    async_result = await parser_instance.aparse(test_message)

    assert sync_result == async_result == expected_result

    # Test invoke methods too
    sync_invoke_result = parser_instance.invoke(test_message)
    async_invoke_result = await parser_instance.ainvoke(test_message)

    assert sync_invoke_result == async_invoke_result == expected_result


@pytest.mark.parametrize(
    ('parser_instance', 'test_message', 'expected_result'),
    [
        (StrOutputParser(), AIMessage(content='consistent'), 'consistent'),
        (JsonOutputParser(), AIMessage(content='{"test": true}'), {'test': True}),
        (YamlOutputParser(), AIMessage(content='test: true'), {'test': True}),
        (
            FenceOutputParser('python'),
            AIMessage(content='```python\nconsistent_result = "test"\n```'),
            'consistent_result = "test"',
        ),
    ],
)
def test_parser_state_consistency(
    parser_instance: BaseOutputParser[Any], test_message: AIMessage, expected_result: Any
) -> None:
    """Test that parsers maintain consistent behavior across multiple calls."""
    # Multiple calls should produce identical results
    result1 = parser_instance.parse(test_message)
    result2 = parser_instance.parse(test_message)
    result3 = parser_instance.parse(test_message)

    assert result1 == result2 == result3 == expected_result


# --- FenceOutputParser Tests --- #


@pytest.mark.parametrize(
    ('raw_content', 'expected_parsed'),
    [
        # Basic fenced blocks with language tag
        ('```python\nprint("hello")\n```', 'print("hello")'),
        ('```PYTHON\nprint("hello")\n```', 'print("hello")'),  # Case insensitive
        ('```Python\nprint("hello")\n```', 'print("hello")'),  # Mixed case
        # Fenced blocks without language tag (enhanced relaxed mode extracts content)
        ('```\nprint("hello")\n```', 'print("hello")'),
        # Multi-line content
        (
            '```python\ndef hello():\n    print("world")\n    return True\n```',
            'def hello():\n    print("world")\n    return True',
        ),
        # With extra whitespace and newlines
        ('  ```python\n  print("hello")  \n```  ', 'print("hello")'),
        (' \n ```python\n print("test") \n``` \n ', 'print("test")'),
        # Missing closing fence
        ('```python\nprint("hello")', 'print("hello")'),
        ('```python\nprint("hello")\nprint("world")', 'print("hello")\nprint("world")'),
        # No fences at all (fallback to whole content)
        ('print("hello")', 'print("hello")'),
        ('def func():\n    pass', 'def func():\n    pass'),
        # Empty content cases
        ('```python\n```', ''),
        ('```python\n\n```', ''),
        ('  \n  ', ''),
        ('', ''),
        # Content with other language tags (enhanced relaxed mode extracts content)
        ('```javascript\nconsole.log("hello")\n```', 'console.log("hello")'),
        ('```sql\nSELECT * FROM users\n```', 'SELECT * FROM users'),
        # Mixed content with correct language tag
        ('Here is the code:\n```python\nprint("hello")\n```\nThat\'s it!', 'print("hello")'),
        ('Some text\n\n```python\ndef test():\n    return 42\n```\n\nMore text', 'def test():\n    return 42'),
        # Edge cases with special characters
        ('```python\nprint("Hello, \\"world\\"!")\n```', 'print("Hello, \\"world\\"!")'),
        (
            '```python\n# Comment with special chars: @#$%^&*()\nprint("test")\n```',
            '# Comment with special chars: @#$%^&*()\nprint("test")',
        ),
        # Nested backticks in content (actually works correctly)
        ('```python\nprint("```")\n```', 'print("```")'),
        ('```python\ncode = "```python\\nprint()\\n```"\n```', 'code = "```python\\nprint()\\n```"'),
    ],
)
def test_python_fenced_output_parser_parse_success(
    python_fenced_parser: BaseOutputParser[str], raw_content: str, expected_parsed: str
) -> None:
    """Test successful Python fenced parsing with various formats."""
    message = AIMessage(content=raw_content)
    assert python_fenced_parser.parse(message) == expected_parsed


@pytest.mark.parametrize(
    ('raw_content', 'expected_parsed'),
    [
        # Should match any language tag
        ('```python\nprint("hello")\n```', 'print("hello")'),
        ('```javascript\nconsole.log("hello")\n```', 'console.log("hello")'),
        ('```sql\nSELECT * FROM users\n```', 'SELECT * FROM users'),
        ('```bash\necho "hello"\n```', 'echo "hello"'),
        ('```\nsome content\n```', 'some content'),
        # Language tags with numbers/special chars
        ('```python3\nprint("hello")\n```', 'print("hello")'),
        ('```c++\nstd::cout << "hello";\n```', 'std::cout << "hello";'),
        ('```objective-c\nNSLog(@"hello");\n```', 'NSLog(@"hello");'),
        # No language tag
        ('```\ngeneric content\n```', 'generic content'),
        # Unfenced content (fallback)
        ('just plain text', 'just plain text'),
        ('multi\nline\ntext', 'multi\nline\ntext'),
        # Mixed content - should extract first fenced block
        ('Text before\n```python\ncode here\n```\nText after', 'code here'),
        ('```js\nfirst\n```\n```python\nsecond\n```', 'first'),  # First match wins
    ],
)
def test_generic_fenced_output_parser_parse_success(
    generic_fenced_parser: BaseOutputParser[str], raw_content: str, expected_parsed: str
) -> None:
    """Test successful generic fenced parsing with various formats."""
    message = AIMessage(content=raw_content)
    assert generic_fenced_parser.parse(message) == expected_parsed


@pytest.mark.parametrize(
    ('invalid_input', 'expected_exception'),
    [
        # These should fail in the parent StrOutputParser first
        (NotAMessage(data='test'), TypeError),
        (123, TypeError),
        (None, TypeError),
        # This has content, but it's not a string
        (BadDuckMessage(content=123), ValueError),
    ],
)
def test_fenced_output_parser_parse_failure_str_parser(
    python_fenced_parser: BaseOutputParser[str], invalid_input: Any, expected_exception: type[BaseException]
) -> None:
    """Test that errors from the parent StrOutputParser are propagated."""
    with pytest.raises(expected_exception):
        python_fenced_parser.parse(invalid_input)


def test_fenced_output_parser_factory_different_types() -> None:
    """Test that the factory creates different parser instances for different fence types."""
    python_parser = FenceOutputParser('python')
    js_parser = FenceOutputParser('javascript')
    generic_parser = FenceOutputParser()

    # They should be different instances
    assert python_parser != js_parser
    assert python_parser != generic_parser
    assert js_parser != generic_parser

    # Test that they parse their respective languages correctly
    python_content = AIMessage(content='```python\nprint("hello")\n```')
    js_content = AIMessage(content='```javascript\nconsole.log("hello")\n```')

    assert python_parser.parse(python_content) == 'print("hello")'
    assert js_parser.parse(js_content) == 'console.log("hello")'
    assert generic_parser.parse(python_content) == 'print("hello")'
    assert generic_parser.parse(js_content) == 'console.log("hello")'


def test_fenced_output_parser_special_fence_types() -> None:
    """Test fenced parsers with special characters in fence types."""
    # Test with fence types that need escaping
    cpp_parser = FenceOutputParser('c++')
    csharp_parser = FenceOutputParser('c#')
    objc_parser = FenceOutputParser('objective-c')

    # Test that they correctly match their specific language tags
    cpp_content = AIMessage(content='```c++\nstd::cout << "hello";\n```')
    csharp_content = AIMessage(content='```c#\nConsole.WriteLine("hello");\n```')
    objc_content = AIMessage(content='```objective-c\nNSLog(@"hello");\n```')

    assert cpp_parser.parse(cpp_content) == 'std::cout << "hello";'
    assert csharp_parser.parse(csharp_content) == 'Console.WriteLine("hello");'
    assert objc_parser.parse(objc_content) == 'NSLog(@"hello");'

    # Test that they don't match other languages (enhanced relaxed mode extracts content)
    wrong_content = AIMessage(content='```python\nprint("hello")\n```')
    assert cpp_parser.parse(wrong_content) == 'print("hello")'


def test_fenced_output_parser_empty_fence_type() -> None:
    """Test that empty fence type creates a generic parser."""
    empty_parser = FenceOutputParser('')
    generic_parser = FenceOutputParser()

    test_content = AIMessage(content='```python\nprint("hello")\n```')

    # Both should behave the same way
    assert empty_parser.parse(test_content) == 'print("hello")'
    assert generic_parser.parse(test_content) == 'print("hello")'


# --- Mode Parameter Tests --- #


def test_fenced_output_parser_invalid_mode() -> None:
    """Test that invalid mode raises ValueError."""
    from typing import cast

    with pytest.raises(ValueError, match="Invalid mode 'invalid'"):
        FenceOutputParser('python', mode=cast('Literal["relaxed", "strict", "typed_strict"]', 'invalid'))


@pytest.mark.parametrize(
    ('mode', 'content', 'expected_result', 'should_raise'),
    [
        # Relaxed mode (enhanced) - tries specific, then generic, then unfenced
        ('relaxed', '```python\nprint("hello")\n```', 'print("hello")', False),
        ('relaxed', '```javascript\nconsole.log("hello")\n```', 'console.log("hello")', False),
        ('relaxed', '```\ngeneric code\n```', 'generic code', False),
        ('relaxed', 'unfenced code', 'unfenced code', False),
        # Strict mode - requires fences but any language
        ('strict', '```python\nprint("hello")\n```', 'print("hello")', False),
        ('strict', '```javascript\nconsole.log("hello")\n```', 'console.log("hello")', False),
        ('strict', '```\ngeneric code\n```', 'generic code', False),
        ('strict', 'unfenced code', None, True),  # Should raise ParseError
        # Typed strict mode - requires specific language fence
        ('typed_strict', '```python\nprint("hello")\n```', 'print("hello")', False),
        ('typed_strict', '```javascript\nconsole.log("hello")\n```', None, True),  # Wrong language
        ('typed_strict', '```\ngeneric code\n```', None, True),  # No language tag
        ('typed_strict', 'unfenced code', None, True),  # No fence
    ],
)
def test_fenced_output_parser_modes_python(
    mode: Literal['relaxed', 'strict', 'typed_strict'], content: str, expected_result: str | None, should_raise: bool
) -> None:
    """Test different parsing modes with Python-specific parser."""
    parser = FenceOutputParser('python', mode=mode)
    message = AIMessage(content=content)

    if should_raise:
        with pytest.raises(ParseError):
            parser.parse(message)
    else:
        assert parser.parse(message) == expected_result


@pytest.mark.parametrize(
    ('mode', 'content', 'expected_result', 'should_raise'),
    [
        # Relaxed mode with generic parser
        ('relaxed', '```python\nprint("hello")\n```', 'print("hello")', False),
        ('relaxed', '```javascript\nconsole.log("hello")\n```', 'console.log("hello")', False),
        ('relaxed', '```\ngeneric code\n```', 'generic code', False),
        ('relaxed', 'unfenced code', 'unfenced code', False),
        # Strict mode with generic parser
        ('strict', '```python\nprint("hello")\n```', 'print("hello")', False),
        ('strict', '```javascript\nconsole.log("hello")\n```', 'console.log("hello")', False),
        ('strict', '```\ngeneric code\n```', 'generic code', False),
        ('strict', 'unfenced code', None, True),  # Should raise ParseError
        # Typed strict mode with generic parser (no specific type, so matches any)
        ('typed_strict', '```python\nprint("hello")\n```', 'print("hello")', False),
        ('typed_strict', '```javascript\nconsole.log("hello")\n```', 'console.log("hello")', False),
        ('typed_strict', '```\ngeneric code\n```', 'generic code', False),
        ('typed_strict', 'unfenced code', None, True),  # No fence
    ],
)
def test_fenced_output_parser_modes_generic(
    mode: Literal['relaxed', 'strict', 'typed_strict'], content: str, expected_result: str | None, should_raise: bool
) -> None:
    """Test different parsing modes with generic parser (no fence_type)."""
    parser = FenceOutputParser('', mode=mode)
    message = AIMessage(content=content)

    if should_raise:
        with pytest.raises(ParseError):
            parser.parse(message)
    else:
        assert parser.parse(message) == expected_result


def test_fenced_output_parser_mode_error_messages() -> None:
    """Test that error messages are informative for different modes."""
    python_parser = FenceOutputParser('python', mode='typed_strict')
    strict_parser = FenceOutputParser('python', mode='strict')

    # Test typed_strict error message
    with pytest.raises(ParseError) as exc_info:
        python_parser.parse(AIMessage(content='```javascript\nconsole.log("hello")\n```'))
    assert "No fenced block with language 'python' found" in str(exc_info.value)

    # Test strict mode error message
    with pytest.raises(ParseError) as exc_info:
        strict_parser.parse(AIMessage(content='unfenced code'))
    assert 'No fenced content found in strict mode' in str(exc_info.value)


def test_fenced_output_parser_mode_edge_cases() -> None:
    """Test edge cases with different modes."""
    parser = FenceOutputParser('python', mode='typed_strict')

    # Empty fenced block should work
    assert parser.parse(AIMessage(content='```python\n```')) == ''

    # Fenced block with just whitespace should work
    assert parser.parse(AIMessage(content='```python\n   \n```')) == ''

    # Mixed content with correct fence should extract the fenced part
    mixed_content = 'Some text\n```python\nprint("hello")\n```\nMore text'
    assert parser.parse(AIMessage(content=mixed_content)) == 'print("hello")'


@pytest.mark.asyncio
async def test_fenced_output_parser_modes_async() -> None:
    """Test that modes work correctly with async methods."""
    parser = FenceOutputParser('python', mode='typed_strict')

    # Should work with correct fence
    result = await parser.aparse(AIMessage(content='```python\nprint("async")\n```'))
    assert result == 'print("async")'

    # Should raise error with wrong fence
    with pytest.raises(ParseError):
        await parser.aparse(AIMessage(content='```javascript\nconsole.log("async")\n```'))

    # Should raise error with no fence
    with pytest.raises(ParseError):
        await parser.aparse(AIMessage(content='unfenced code'))


def test_fenced_output_parser_mode_consistency() -> None:
    """Test that mode behavior is consistent across multiple calls."""
    parsers = [
        (FenceOutputParser('python', mode='relaxed'), 'relaxed'),
        (FenceOutputParser('python', mode='strict'), 'strict'),
        (FenceOutputParser('python', mode='typed_strict'), 'typed_strict'),
    ]

    test_cases = [
        ('```python\ncode\n```', True, True, True),  # All should work
        ('```javascript\ncode\n```', True, True, False),  # Only relaxed and strict
        ('unfenced code', True, False, False),  # Only relaxed
    ]

    for content, relaxed_works, strict_works, typed_strict_works in test_cases:
        expected_results = [relaxed_works, strict_works, typed_strict_works]

        for (parser, _mode_name), should_work in zip(parsers, expected_results, strict=False):
            message = AIMessage(content=content)

            if should_work:
                # Should not raise an exception
                result = parser.parse(message)
                assert isinstance(result, str)
            else:
                # Should raise ParseError
                with pytest.raises(ParseError):
                    parser.parse(message)


# --- Async Tests for FenceOutputParser --- #


@pytest.mark.parametrize(
    ('input_data', 'expected_result', 'should_raise', 'expected_exception'),
    [
        ('```python\nresult = True\nprint(result)\n```', 'result = True\nprint(result)', False, TypeError),
        ('plain python code without fences', 'plain python code without fences', False, TypeError),
        (NotAMessage(data='test'), None, True, TypeError),
    ],
)
@pytest.mark.asyncio
async def test_fenced_output_parser_aparse(
    python_fenced_parser: BaseOutputParser[str],
    input_data: Any,
    expected_result: str | None,
    should_raise: bool,
    expected_exception: type[Exception],
) -> None:
    """Test async fenced parsing with various scenarios."""
    if should_raise:
        with pytest.raises(expected_exception):
            await python_fenced_parser.aparse(input_data)
    else:
        message = AIMessage(content=input_data) if isinstance(input_data, str) else input_data
        result = await python_fenced_parser.aparse(message)
        assert result == expected_result


# --- Invoke/AInvoke Tests for FenceOutputParser --- #


@pytest.mark.parametrize(
    ('parser_type', 'content', 'expected_result'),
    [
        ('python', '```python\ndata = [1, 2, 3]\nprint(data)\n```', 'data = [1, 2, 3]\nprint(data)'),
        ('python', 'x = 42', 'x = 42'),
        ('python', '```python\nfor i in range(3):\n    print(i)\n```', 'for i in range(3):\n    print(i)'),
        ('generic', '```sql\nSELECT COUNT(*) FROM users;\n```', 'SELECT COUNT(*) FROM users;'),
    ],
)
def test_fenced_output_parser_invoke(parser_type: str, content: str, expected_result: str) -> None:
    """Test sync invocation with various content types."""
    parser = FenceOutputParser('python') if parser_type == 'python' else FenceOutputParser()
    message = AIMessage(content=content)
    assert parser.invoke(message) == expected_result


@pytest.mark.parametrize(
    ('parser_type', 'content', 'expected_result'),
    [
        ('python', '```python\ndata = [1, 2, 3]\nprint(data)\n```', 'data = [1, 2, 3]\nprint(data)'),
        ('python', 'x = 42', 'x = 42'),
        ('python', '```python\nfor i in range(3):\n    print(i)\n```', 'for i in range(3):\n    print(i)'),
        ('generic', '```sql\nSELECT COUNT(*) FROM users;\n```', 'SELECT COUNT(*) FROM users;'),
    ],
)
@pytest.mark.asyncio
async def test_fenced_output_parser_ainvoke(parser_type: str, content: str, expected_result: str) -> None:
    """Test async invocation with various content types."""
    parser = FenceOutputParser('python') if parser_type == 'python' else FenceOutputParser()
    message = AIMessage(content=content)
    assert await parser.ainvoke(message) == expected_result


# --- Interface Contract Tests for FenceOutputParser --- #


# FenceOutputParser interface contract and consistency tests are now covered by the consolidated
# parametrized tests above


# --- Edge Case and Robustness Tests --- #


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        # Whitespace handling
        ('```python\n   print("hello")   \n```', 'print("hello")'),
        ('```python\n\tprint("hello")\n\t\n```', 'print("hello")'),
        ('```python\n\n\nprint("hello")\n\n\n```', 'print("hello")'),
        ('   ```python\nprint("hello")\n```   ', 'print("hello")'),
        ('```python\n   \n\t\n   \n```', ''),
        # Malformed fences
        ('``python\nprint("hello")\n```', '``python\nprint("hello")\n```'),  # Fallback
        ('````python\nprint("hello")\n````', 'print("hello")'),  # Actually matches
        ('text ```python\nprint("hello")\n``` more text', 'print("hello")'),  # Should still match
        ('```python\nfirst\n```\n```python\nsecond\n```', 'first'),  # First match
        ('```python\nprint("```")\nprint("code")\n```', 'print("```")\nprint("code")'),
        # Unicode and special characters
        ('```python\nprint("Hello, 世界!")\n```', 'print("Hello, 世界!")'),
        ('```python\nprint("🐍 Python!")\n```', 'print("🐍 Python!")'),
        (
            '```python\nprint("Special: @#$%^&*()_+-={}[]|\\:;"\'<>?,./")\n```',
            'print("Special: @#$%^&*()_+-={}[]|\\:;"\'<>?,./")',
        ),
    ],
)
def test_fenced_output_parser_edge_cases(content: str, expected: str) -> None:
    """Test various edge cases including whitespace, malformed fences, and special characters."""
    parser = FenceOutputParser('python')
    message = AIMessage(content=content)
    assert parser.parse(message) == expected


def test_fenced_output_parser_large_content() -> None:
    """Test handling of large content blocks."""
    parser = FenceOutputParser('python')

    # Generate large content
    large_code = '\n'.join([f'print("Line {i}")' for i in range(1000)])
    content = f'```python\n{large_code}\n```'

    message = AIMessage(content=content)
    result = parser.parse(message)

    assert result == large_code
    assert len(result.split('\n')) == 1000


# --- Multiple Fences Bug Tests --- #


def test_fenced_output_parser_multiple_fences_priority() -> None:
    """Test that specific fence types are prioritized over generic ones."""
    # Test with Python parser - should prioritize Python fences
    python_parser = FenceOutputParser('python', mode='relaxed')

    # Python fence first - should pick Python
    content1 = '```python\nprint("python")\n```\n```javascript\nconsole.log("js")\n```'
    assert python_parser.parse(AIMessage(content=content1)) == 'print("python")'

    # JavaScript fence first, Python second - should pick Python (specific match)
    content2 = '```javascript\nconsole.log("js")\n```\n```python\nprint("python")\n```'
    assert python_parser.parse(AIMessage(content=content2)) == 'print("python")'

    # No Python fence, only JavaScript - should extract JavaScript content (enhanced relaxed)
    content3 = '```javascript\nconsole.log("js")\n```\n```sql\nSELECT * FROM users\n```'
    assert python_parser.parse(AIMessage(content=content3)) == 'console.log("js")'


def test_fenced_output_parser_enhanced_relaxed_mode() -> None:
    """Test the enhanced relaxed mode fallback chain."""
    parser = FenceOutputParser('python', mode='relaxed')

    # 1. Specific fence match (highest priority)
    content1 = '```python\nprint("specific")\n```'
    assert parser.parse(AIMessage(content=content1)) == 'print("specific")'

    # 2. Generic fence match (middle priority)
    content2 = '```javascript\nconsole.log("generic")\n```'
    assert parser.parse(AIMessage(content=content2)) == 'console.log("generic")'

    # 3. Unfenced content (lowest priority)
    content3 = 'plain text content'
    assert parser.parse(AIMessage(content=content3)) == 'plain text content'

    # Mixed: specific fence should win over generic
    content4 = '```javascript\nconsole.log("js")\n```\n```python\nprint("python")\n```'
    assert parser.parse(AIMessage(content=content4)) == 'print("python")'


# --- Additional Corner Cases from Discussion --- #


def test_fenced_output_parser_empty_fence_edge_cases() -> None:
    """Test specific empty fence edge cases we discovered and fixed."""
    parser = FenceOutputParser('python')

    # Edge case: content is just the closing fence (should return empty string)
    content1 = '```python\n```\n```'
    assert parser.parse(AIMessage(content=content1)) == ''

    # Edge case: content contains only closing fence markers
    content2 = '```python\n```'
    assert parser.parse(AIMessage(content=content2)) == ''

    # Edge case: whitespace around closing fence
    content3 = '```python\n  ```  \n```'
    assert parser.parse(AIMessage(content=content3)) == ''


def test_json_yaml_multiple_fences_scenarios() -> None:
    """Test JSON and YAML parsers with multiple fences scenarios we discussed."""
    json_parser = JsonOutputParser()
    yaml_parser = YamlOutputParser()

    # JSON parser should prioritize JSON fences over others
    content1 = '```python\nprint("hello")\n```\n```json\n{"key": "value"}\n```'
    assert json_parser.parse(AIMessage(content=content1)) == {'key': 'value'}

    # YAML parser should prioritize YAML fences over others
    content2 = '```sql\nSELECT * FROM users\n```\n```yaml\nkey: value\n```'
    assert yaml_parser.parse(AIMessage(content=content2)) == {'key': 'value'}

    # JSON parser with no JSON fence should extract first fence content (enhanced relaxed)
    content3 = '```python\n{"not_python": true}\n```\n```sql\n{"also_not_sql": true}\n```'
    assert json_parser.parse(AIMessage(content=content3)) == {'not_python': True}

    # YAML parser with no YAML fence should extract first fence content (enhanced relaxed)
    content4 = '```javascript\nkey: value\n```\n```bash\nanother: item\n```'
    assert yaml_parser.parse(AIMessage(content=content4)) == {'key': 'value'}


def test_fenced_output_parser_regex_edge_cases() -> None:
    """Test regex edge cases we encountered during development."""
    parser = FenceOutputParser('python')

    # Edge case: fence with extra characters after language tag
    content1 = '```python extra stuff\nprint("hello")\n```'
    assert parser.parse(AIMessage(content=content1)) == 'print("hello")'

    # Edge case: fence with numbers and special chars in language tag
    content2 = '```python3.11\nprint("hello")\n```'
    assert parser.parse(AIMessage(content=content2)) == 'print("hello")'

    # Edge case: multiple closing fences
    content3 = '```python\nprint("hello")\n```\n```\n```'
    assert parser.parse(AIMessage(content=content3)) == 'print("hello")'


def test_fenced_output_parser_case_sensitivity_edge_cases() -> None:
    """Test case sensitivity edge cases for language tags."""
    # Test various case combinations
    test_cases = [
        ('python', '```PYTHON\ncode\n```', 'code'),
        ('Python', '```python\ncode\n```', 'code'),  # Should match case-insensitively
        ('PYTHON', '```python\ncode\n```', 'code'),  # Should match case-insensitively
        ('c++', '```C++\ncode\n```', 'code'),
        ('C++', '```c++\ncode\n```', 'code'),
    ]

    for fence_type, content, expected in test_cases:
        parser = FenceOutputParser(fence_type)
        assert parser.parse(AIMessage(content=content)) == expected


def test_fenced_output_parser_boundary_conditions() -> None:
    """Test boundary conditions and extreme cases."""
    parser = FenceOutputParser('python')

    # Very long language tag
    long_lang = 'python' + 'x' * 100
    content1 = f'```{long_lang}\ncode\n```'
    # Should fall back to generic fence extraction in enhanced relaxed mode
    assert parser.parse(AIMessage(content=content1)) == 'code'

    # Empty language tag with spaces
    content2 = '```   \ncode\n```'
    assert parser.parse(AIMessage(content=content2)) == 'code'

    # Language tag with only special characters
    content3 = '```@#$%\ncode\n```'
    assert parser.parse(AIMessage(content=content3)) == 'code'


def test_fenced_output_parser_newline_variations() -> None:
    """Test different newline variations we encountered."""
    parser = FenceOutputParser('python')

    # Windows-style newlines
    content1 = '```python\r\nprint("hello")\r\n```'
    assert parser.parse(AIMessage(content=content1)) == 'print("hello")'

    # Mixed newlines
    content2 = '```python\nprint("hello")\r\nprint("world")\n```'
    assert parser.parse(AIMessage(content=content2)) == 'print("hello")\r\nprint("world")'

    # No newline after opening fence
    content3 = '```pythonprint("hello")\n```'
    # Our regex actually matches this as a malformed fence and extracts empty content
    assert parser.parse(AIMessage(content=content3)) == ''


def test_fenced_output_parser_performance_edge_case() -> None:
    """Test the performance edge case with deeply nested content."""
    parser = FenceOutputParser('python')

    # Create content with many nested backticks
    nested_content = 'print("' + '`' * 1000 + '")'
    content = f'```python\n{nested_content}\n```'

    result = parser.parse(AIMessage(content=content))
    assert result == nested_content
