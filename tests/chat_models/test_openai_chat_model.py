"""Tests for the ChatOpenAI chat model implementation."""

import asyncio
from typing import Any
from typing import Generator
from dataclasses import field
from unittest.mock import Mock
from unittest.mock import patch
from dataclasses import dataclass
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import openai
import pytest
from openai import NOT_GIVEN
from pydantic import BaseModel
from openai import DEFAULT_MAX_RETRIES

from tinylcel.messages import AIMessage
from tinylcel.messages import BaseMessage
from tinylcel.messages import HumanMessage
from tinylcel.messages import MessagesInput
from tinylcel.messages import SystemMessage
from tinylcel.providers.openai import ChatOpenAI
from tinylcel.providers.openai import AzureChatOpenAI
from tinylcel.providers.openai.chat_models import StructureChatOpenAI


# Helper dataclass for testing unknown roles
@dataclass(frozen=True)
class SpecialRoleMessage(BaseMessage):
    """Special message type for testing non-standard roles."""

    role: str = field(default='special_role', init=False)


# Test Pydantic models for structured output testing
class Person(BaseModel):
    """Test model for structured output."""

    name: str
    age: int
    city: str


class SimpleResponse(BaseModel):
    """Simple test model for structured output."""

    message: str
    success: bool


# Constants for Azure tests
TEST_AZURE_API_KEY = 'test-azure-api-key'
TEST_AZURE_ENDPOINT = 'https://test-azure-endpoint.openai.azure.com'
TEST_AZURE_API_VERSION = '2023-07-01-preview'
TEST_AZURE_DEPLOYMENT_NAME = 'test-deployment'
TEST_AZURE_MODEL_NAME = 'gpt-35-turbo-test'
TEST_OPENAI_API_KEY = 'test-openai-key'


@pytest.fixture
def mock_openai_clients() -> Generator[dict[str, Any], None, None]:
    """Fixture to mock openai.OpenAI and openai.AsyncOpenAI clients."""
    mock_sync_client = MagicMock(spec=openai.OpenAI)
    mock_sync_completions = MagicMock()
    mock_sync_chat = MagicMock()
    mock_sync_client.chat = mock_sync_chat
    mock_sync_chat.completions = mock_sync_completions
    mock_sync_response = MagicMock()
    mock_sync_choice = MagicMock()
    mock_sync_message = MagicMock()
    mock_sync_message.content = 'Mocked sync response'
    mock_sync_choice.message = mock_sync_message
    mock_sync_response.choices = [mock_sync_choice]
    mock_sync_completions.create.return_value = mock_sync_response

    mock_async_response = MagicMock()
    mock_async_choice = MagicMock()
    mock_async_message = MagicMock()
    mock_async_message.content = 'Mocked async response'
    mock_async_choice.message = mock_async_message
    mock_async_response.choices = [mock_async_choice]
    mock_async_create_method = AsyncMock(return_value=mock_async_response)

    with (
        patch(
            'tinylcel.providers.openai.chat_models.openai.OpenAI', return_value=mock_sync_client
        ) as mock_sync_constructor,
        patch('tinylcel.providers.openai.chat_models.openai.AsyncOpenAI') as mock_async_constructor_patch,
    ):
        mock_async_client_instance = MagicMock()
        mock_async_completions_instance = MagicMock()
        mock_async_chat_instance = MagicMock()
        mock_async_client_instance.chat = mock_async_chat_instance
        mock_async_chat_instance.completions = mock_async_completions_instance
        mock_async_completions_instance.create = mock_async_create_method
        mock_async_constructor_patch.return_value = mock_async_client_instance
        yield {
            'sync_client': mock_sync_client,
            'async_client': mock_async_client_instance,
            'sync_create': mock_sync_completions.create,
            'async_create': mock_async_create_method,
            'sync_constructor': mock_sync_constructor,
            'async_constructor': mock_async_constructor_patch,
        }


@pytest.fixture
def mock_azure_clients() -> Generator[dict[str, Any], None, None]:
    """Fixture to mock openai.AzureOpenAI and openai.AsyncAzureOpenAI clients."""
    mock_sync_azure_client = MagicMock(spec=openai.AzureOpenAI)
    mock_sync_azure_completions = MagicMock()
    mock_sync_azure_chat = MagicMock()
    mock_sync_azure_client.chat = mock_sync_azure_chat
    mock_sync_azure_chat.completions = mock_sync_azure_completions
    mock_sync_azure_response = MagicMock()
    mock_sync_azure_choice = MagicMock()
    mock_sync_azure_message = MagicMock()
    mock_sync_azure_message.content = 'Mocked Azure sync response'
    mock_sync_azure_choice.message = mock_sync_azure_message
    mock_sync_azure_response.choices = [mock_sync_azure_choice]
    mock_sync_azure_completions.create.return_value = mock_sync_azure_response

    mock_async_azure_response = MagicMock()
    mock_async_azure_choice = MagicMock()
    mock_async_azure_message = MagicMock()
    mock_async_azure_message.content = 'Mocked Azure async response'
    mock_async_azure_choice.message = mock_async_azure_message
    mock_async_azure_response.choices = [mock_async_azure_choice]
    mock_async_azure_create_method = AsyncMock(return_value=mock_async_azure_response)

    with (
        patch(
            'tinylcel.providers.openai.chat_models.AzureOpenAI', return_value=mock_sync_azure_client
        ) as mock_sync_azure_constructor,
        patch('tinylcel.providers.openai.chat_models.AsyncAzureOpenAI') as mock_async_azure_constructor_patch,
    ):
        mock_async_azure_client_instance = MagicMock(spec=openai.AsyncAzureOpenAI)
        mock_async_azure_completions_instance = MagicMock()
        mock_async_azure_chat_instance = MagicMock()
        mock_async_azure_client_instance.chat = mock_async_azure_chat_instance
        mock_async_azure_chat_instance.completions = mock_async_azure_completions_instance
        mock_async_azure_completions_instance.create = mock_async_azure_create_method
        mock_async_azure_constructor_patch.return_value = mock_async_azure_client_instance
        yield {
            'sync_client': mock_sync_azure_client,
            'async_client': mock_async_azure_client_instance,
            'sync_create': mock_sync_azure_completions.create,
            'async_create': mock_async_azure_create_method,
            'sync_constructor': mock_sync_azure_constructor,
            'async_constructor': mock_async_azure_constructor_patch,
        }


@pytest.fixture
def sample_messages() -> MessagesInput:
    return [
        SystemMessage(content='You are a helpful assistant.'),
        HumanMessage(content='What is the capital of France?'),
    ]


@pytest.fixture
def sample_messages_dict_openai_fmt() -> list[dict[str, str]]:
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the capital of France?'},
    ]


@pytest.fixture
def mock_structured_output_clients() -> Generator[dict[str, Any], None, None]:
    """Fixture to mock openai clients for structured output (beta.chat.completions.parse)."""
    # Mock sync client
    mock_sync_client = MagicMock(spec=openai.OpenAI)
    mock_sync_beta = MagicMock()
    mock_sync_chat = MagicMock()
    mock_sync_completions = MagicMock()
    mock_sync_client.beta = mock_sync_beta
    mock_sync_beta.chat = mock_sync_chat
    mock_sync_chat.completions = mock_sync_completions

    # Mock sync parse response
    mock_sync_parse_response = MagicMock()
    mock_sync_parse_response.choices = [MagicMock()]
    mock_sync_parse_response.choices[0].message = MagicMock()
    # Set up the parsed response - will be set by individual tests
    mock_sync_parse_response.choices[0].message.parsed = None
    mock_sync_completions.parse.return_value = mock_sync_parse_response

    # Mock async client
    mock_async_client = MagicMock(spec=openai.AsyncOpenAI)
    mock_async_beta = MagicMock()
    mock_async_chat = MagicMock()
    mock_async_completions = MagicMock()
    mock_async_client.beta = mock_async_beta
    mock_async_beta.chat = mock_async_chat
    mock_async_chat.completions = mock_async_completions

    # Mock async parse response
    mock_async_parse_response = MagicMock()
    mock_async_parse_response.choices = [MagicMock()]
    mock_async_parse_response.choices[0].message = MagicMock()
    mock_async_parse_response.choices[0].message.parsed = None
    mock_async_parse_method = AsyncMock(return_value=mock_async_parse_response)
    mock_async_completions.parse = mock_async_parse_method

    with (
        patch(
            'tinylcel.providers.openai.chat_models.openai.OpenAI', return_value=mock_sync_client
        ) as mock_sync_constructor,
        patch(
            'tinylcel.providers.openai.chat_models.openai.AsyncOpenAI', return_value=mock_async_client
        ) as mock_async_constructor,
    ):
        yield {
            'sync_client': mock_sync_client,
            'async_client': mock_async_client,
            'sync_parse': mock_sync_completions.parse,
            'async_parse': mock_async_parse_method,
            'sync_parse_response': mock_sync_parse_response,
            'async_parse_response': mock_async_parse_response,
            'sync_constructor': mock_sync_constructor,
            'async_constructor': mock_async_constructor,
        }


@pytest.fixture(autouse=True)
def set_env_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'key-from-env-for-chat-tests')


# --- Test Cases for ChatOpenAI --- #


def test_chatopenai_initialization_defaults(mock_openai_clients: dict[str, Any]) -> None:
    chat_model = ChatOpenAI()
    assert chat_model.model == 'gpt-3.5-turbo'
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_tokens is NOT_GIVEN
    assert chat_model.max_completion_tokens is NOT_GIVEN
    assert chat_model.max_retries == DEFAULT_MAX_RETRIES
    assert chat_model.timeout is NOT_GIVEN
    assert chat_model.api_key is None

    expected_client_kwargs = {
        'api_key': 'key-from-env-for-chat-tests',
        'max_retries': DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
    }
    mock_openai_clients['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_chatopenai_initialization_explicit_key(mock_openai_clients: dict[str, Any]) -> None:
    chat_model = ChatOpenAI(api_key=TEST_OPENAI_API_KEY)
    assert chat_model.api_key == TEST_OPENAI_API_KEY
    expected_client_kwargs = {'api_key': TEST_OPENAI_API_KEY, 'max_retries': DEFAULT_MAX_RETRIES, 'timeout': NOT_GIVEN}
    mock_openai_clients['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_chatopenai_initialization_custom_params(mock_openai_clients: dict[str, Any]) -> None:
    custom_timeout_int = 30
    chat_model = ChatOpenAI(
        model='gpt-4',
        temperature=0.1,
        max_tokens=150,
        max_completion_tokens=100,
        api_key='custom-key',
        max_retries=5,
        timeout=custom_timeout_int,
    )
    assert chat_model.model == 'gpt-4'
    assert chat_model.temperature == 0.1
    assert chat_model.max_tokens == 150
    assert chat_model.max_completion_tokens == 100
    assert chat_model.max_retries == 5
    assert chat_model.timeout == custom_timeout_int
    expected_client_kwargs = {'api_key': 'custom-key', 'max_retries': 5, 'timeout': custom_timeout_int}
    mock_openai_clients['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_chatopenai_init_base_url_is_used_by_clients(mock_openai_clients: dict[str, Any]) -> None:
    """Test that base_url is passed to OpenAI client constructors."""
    # The autouse set_env_api_keys fixture will set OPENAI_API_KEY to "key-from-env-for-chat-tests"
    # get_api_key will use this if api_key arg is None

    base_url_to_test = 'http://localhost:1234'
    # Instantiate ChatOpenAI, __post_init__ will use get_api_key then call client constructors
    chat_model = ChatOpenAI(base_url=base_url_to_test)  # api_key will be from env via get_api_key
    assert chat_model.base_url == base_url_to_test
    assert chat_model.api_key is None  # Since it was taken from env by get_api_key

    expected_client_kwargs = {
        'api_key': 'key-from-env-for-chat-tests',  # This is what get_api_key would return
        'max_retries': openai.DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
        'base_url': base_url_to_test,
    }
    mock_openai_clients['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_chatopenai_initialization_with_max_tokens_only(mock_openai_clients: dict[str, Any]) -> None:
    chat_model = ChatOpenAI(max_tokens=100, api_key='mt-key')
    assert chat_model.max_tokens == 100
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_completion_tokens is NOT_GIVEN
    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key='mt-key', max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key='mt-key', max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )


def test_chatopenai_initialization_with_max_completion_tokens_only(mock_openai_clients: dict[str, Any]) -> None:
    chat_model = ChatOpenAI(max_completion_tokens=200, api_key='mct-key')
    assert chat_model.max_completion_tokens == 200
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_tokens is NOT_GIVEN
    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key='mct-key', max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key='mct-key', max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )


@pytest.mark.parametrize(
    ('input_message', 'expected_dict'),
    [
        (SystemMessage(content='System prompt'), {'role': 'system', 'content': 'System prompt'}),
        (HumanMessage(content='User query'), {'role': 'user', 'content': 'User query'}),
        (AIMessage(content='Assistant response'), {'role': 'assistant', 'content': 'Assistant response'}),
    ],
)
@pytest.mark.usefixtures('mock_openai_clients')
def test_convert_message_to_dict(input_message: BaseMessage, expected_dict: dict[str, str]) -> None:
    from tinylcel.providers.openai.chat_models import _message_to_dict

    assert _message_to_dict(input_message) == expected_dict


@pytest.mark.usefixtures('mock_openai_clients')
def test_convert_message_to_dict_unknown_role_passthrough() -> None:
    """Test _message_to_dict raises ValueError for unknown roles."""
    from tinylcel.providers.openai.chat_models import _message_to_dict

    # Use the new SpecialRoleMessage
    special_message = SpecialRoleMessage(content='Special content')
    with pytest.raises(ValueError, match='Unsupported message type'):
        _message_to_dict(special_message)


# --- ChatOpenAI Invoke / _generate Tests --- #


def test_chatopenai_generate_success_defaults(
    mock_openai_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    chat_model = ChatOpenAI()
    result = chat_model._generate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == 'Mocked sync response'
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN,
    )


@pytest.mark.asyncio
async def test_chatopenai_agenerate_success_defaults(
    mock_openai_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    chat_model = ChatOpenAI()
    result = await chat_model._agenerate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == 'Mocked async response'
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN,
    )


def test_chatopenai_generate_with_all_optional_params_set(
    mock_openai_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    chat_model = ChatOpenAI(temperature=0.2, max_tokens=50, max_completion_tokens=25)
    result = chat_model._generate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == 'Mocked sync response'
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=0.2,
        max_tokens=50,
        max_completion_tokens=25,
    )


@pytest.mark.asyncio
async def test_chatopenai_agenerate_with_all_optional_params_set(
    mock_openai_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    chat_model = ChatOpenAI(temperature=0.3, max_tokens=60, max_completion_tokens=30)
    result = await chat_model._agenerate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == 'Mocked async response'
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=0.3,
        max_tokens=60,
        max_completion_tokens=30,
    )


@pytest.mark.parametrize(
    ('param_name', 'param_value'),
    [
        ('temperature', 0.11),
        ('max_tokens', 111),
        ('max_completion_tokens', 222),
    ],
)
def test_chatopenai_generate_specific_optional_param(
    mock_openai_clients: dict,
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
    param_name: str,
    param_value: Any,
) -> None:
    """Test _generate with one specific optional parameter set."""
    chat_model_params = {'model': 'gpt-test', param_name: param_value}
    if 'api_key' not in chat_model_params:
        chat_model_params['api_key'] = 'dummy_key_for_specific_param_test'
    chat_model = ChatOpenAI(**chat_model_params)

    chat_model._generate(sample_messages)

    expected_api_kwargs = {
        'model': 'gpt-test',
        'messages': sample_messages_dict_openai_fmt,
        'temperature': NOT_GIVEN,
        'max_tokens': NOT_GIVEN,
        'max_completion_tokens': NOT_GIVEN,
    }
    expected_api_kwargs[param_name] = param_value

    mock_openai_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('param_name', 'param_value'),
    [
        ('temperature', 0.22),
        ('max_tokens', 333),
        ('max_completion_tokens', 444),
    ],
)
async def test_chatopenai_agenerate_specific_optional_param(
    mock_openai_clients: dict,
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
    param_name: str,
    param_value: Any,
) -> None:
    """Test _agenerate with one specific optional parameter set."""
    chat_model_params = {'model': 'gpt-test-async', param_name: param_value}
    if 'api_key' not in chat_model_params:
        chat_model_params['api_key'] = 'dummy_key_for_specific_param_test_async'
    chat_model = ChatOpenAI(**chat_model_params)

    await chat_model._agenerate(sample_messages)

    expected_api_kwargs = {
        'model': 'gpt-test-async',
        'messages': sample_messages_dict_openai_fmt,
        'temperature': NOT_GIVEN,
        'max_tokens': NOT_GIVEN,
        'max_completion_tokens': NOT_GIVEN,
    }
    expected_api_kwargs[param_name] = param_value

    mock_openai_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)


def test_chatopenai_invoke_success_defaults(
    mock_openai_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    chat_model = ChatOpenAI()
    result = chat_model.invoke(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == 'Mocked sync response'
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN,
    )


@pytest.mark.asyncio
async def test_chatopenai_ainvoke_success_defaults(
    mock_openai_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    chat_model = ChatOpenAI()
    result = await chat_model.ainvoke(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == 'Mocked async response'
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN,
    )


def test_chatopenai_generate_api_error(mock_openai_clients: dict[str, Any], sample_messages: MessagesInput) -> None:
    mock_openai_clients['sync_create'].side_effect = openai.APIError('API Failed', request=Mock(), body=None)
    chat_model = ChatOpenAI()
    with pytest.raises(openai.APIError, match='API Failed'):
        chat_model._generate(sample_messages)


def test_chatopenai_generate_none_content(mock_openai_clients: dict[str, Any], sample_messages: MessagesInput) -> None:
    mock_response = mock_openai_clients['sync_create'].return_value
    mock_response.choices[0].message.content = None
    chat_model = ChatOpenAI()
    with pytest.raises(ValueError, match='OpenAI response content is None'):
        chat_model._generate(sample_messages)


@pytest.mark.asyncio
async def test_chatopenai_agenerate_api_error(
    mock_openai_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    mock_openai_clients['async_create'].side_effect = openai.APIError('Async API Failed', request=Mock(), body=None)
    chat_model = ChatOpenAI()
    with pytest.raises(openai.APIError, match='Async API Failed'):
        await chat_model._agenerate(sample_messages)


@pytest.mark.asyncio
async def test_chatopenai_agenerate_none_content(
    mock_openai_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    mock_response = mock_openai_clients['async_create'].return_value
    mock_response.choices[0].message.content = None
    mock_openai_clients['async_create'].side_effect = None
    mock_openai_clients['async_create'].return_value = mock_response

    chat_model = ChatOpenAI()
    with pytest.raises(ValueError, match='OpenAI response content is None'):
        await chat_model._agenerate(sample_messages)


# --- ChatOpenAI Batch Tests --- #
def test_chatopenai_batch_success(mock_openai_clients: dict[str, Any], sample_messages: MessagesInput) -> None:
    chat_model = ChatOpenAI()
    inputs = [sample_messages, sample_messages]
    mock_openai_clients['sync_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content='Sync Resp 1'))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content='Sync Resp 2'))]),
    ]
    results = chat_model.batch(inputs)
    assert len(results) == 2
    assert results[0].content == 'Sync Resp 1'  # type: ignore[union-attr]
    assert results[1].content == 'Sync Resp 2'  # type: ignore[union-attr]
    assert mock_openai_clients['sync_create'].call_count == 2


def test_chatopenai_batch_return_exceptions(
    mock_openai_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    chat_model = ChatOpenAI()
    inputs: list[MessagesInput] = [sample_messages, [HumanMessage(content='error trigger')], sample_messages]
    error = ValueError('Batch Error')
    mock_openai_clients['sync_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content='Sync Resp 1'))]),
        error,
        MagicMock(choices=[MagicMock(message=MagicMock(content='Sync Resp 3'))]),
    ]
    results = chat_model.batch(inputs, return_exceptions=True)
    assert len(results) == 3
    # Type narrowing: check if result is AIMessage before accessing content
    assert isinstance(results[0], AIMessage)
    assert results[0].content == 'Sync Resp 1'  # type: ignore[union-attr]
    assert results[1] is error
    assert isinstance(results[2], AIMessage)
    assert results[2].content == 'Sync Resp 3'  # type: ignore[union-attr]
    assert mock_openai_clients['sync_create'].call_count == 3


@pytest.mark.asyncio
async def test_chatopenai_abatch_success(mock_openai_clients: dict[str, Any], sample_messages: MessagesInput) -> None:
    chat_model = ChatOpenAI()
    inputs = [sample_messages, sample_messages]
    mock_openai_clients['async_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content='Async Resp 1'))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content='Async Resp 2'))]),
    ]
    results = await chat_model.abatch(inputs)
    assert len(results) == 2
    assert results[0].content == 'Async Resp 1'  # type: ignore[union-attr]
    assert results[1].content == 'Async Resp 2'  # type: ignore[union-attr]
    assert mock_openai_clients['async_create'].await_count == 2


@pytest.mark.asyncio
async def test_chatopenai_abatch_return_exceptions(
    mock_openai_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    chat_model = ChatOpenAI()
    inputs: list[MessagesInput] = [sample_messages, [HumanMessage(content='error trigger')], sample_messages]
    error = ValueError('Async Batch Error')
    mock_openai_clients['async_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content='Async Resp 1'))]),
        error,
        MagicMock(choices=[MagicMock(message=MagicMock(content='Async Resp 3'))]),
    ]
    results = await chat_model.abatch(inputs, return_exceptions=True)
    assert len(results) == 3
    # Type narrowing: check if result is AIMessage before accessing content
    assert isinstance(results[0], AIMessage)
    assert results[0].content == 'Async Resp 1'  # type: ignore[union-attr]
    assert results[1] is error
    assert isinstance(results[2], AIMessage)
    assert results[2].content == 'Async Resp 3'  # type: ignore[union-attr]
    assert mock_openai_clients['async_create'].await_count == 3


# --- AzureChatOpenAI Tests (New and existing combined) --- #


def test_azure_chatopenai_initialization_defaults_env_key(
    mock_azure_clients: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    explicit_timeout_int_for_azure = 50
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT, api_version=TEST_AZURE_API_VERSION, timeout=explicit_timeout_int_for_azure
    )
    assert chat_model.azure_endpoint == TEST_AZURE_ENDPOINT
    assert chat_model.api_version == TEST_AZURE_API_VERSION
    assert chat_model.api_key is None
    assert chat_model.azure_deployment is None
    assert chat_model.model == 'gpt-3.5-turbo'
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_tokens is NOT_GIVEN
    assert chat_model.max_completion_tokens is NOT_GIVEN
    assert chat_model.max_retries == DEFAULT_MAX_RETRIES
    assert chat_model.timeout == explicit_timeout_int_for_azure

    expected_client_kwargs = {
        'api_key': TEST_AZURE_API_KEY,
        'azure_endpoint': TEST_AZURE_ENDPOINT,
        'api_version': TEST_AZURE_API_VERSION,
        'max_retries': DEFAULT_MAX_RETRIES,
        'timeout': explicit_timeout_int_for_azure,
        'azure_ad_token': None,
        'azure_ad_token_provider': None,
    }
    mock_azure_clients['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_azure_clients['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_azure_chatopenai_init_minimal_with_env_key(mock_azure_clients: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test AzureChatOpenAI minimal init, API key from env, with deployment."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
    )
    assert chat_model.api_key is None
    assert chat_model.azure_endpoint == TEST_AZURE_ENDPOINT
    assert chat_model.api_version == TEST_AZURE_API_VERSION
    assert chat_model.azure_deployment == TEST_AZURE_DEPLOYMENT_NAME
    assert chat_model.model == 'gpt-3.5-turbo'

    expected_kwargs = {
        'api_key': TEST_AZURE_API_KEY,
        'azure_endpoint': TEST_AZURE_ENDPOINT,
        'api_version': TEST_AZURE_API_VERSION,
        'max_retries': DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
        'azure_ad_token': None,
        'azure_ad_token_provider': None,
    }
    mock_azure_clients['sync_constructor'].assert_called_once_with(**expected_kwargs)
    mock_azure_clients['async_constructor'].assert_called_once_with(**expected_kwargs)


def test_azure_chatopenai_init_explicit_key_deployment_overrides_model(mock_azure_clients: dict) -> None:
    """Test AzureChatOpenAI with explicit key, deployment overrides model."""
    chat_model = AzureChatOpenAI(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        model='original-model-name',
    )
    assert chat_model.api_key == TEST_AZURE_API_KEY
    assert chat_model.model == 'original-model-name'

    mock_azure_clients['sync_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN,
        azure_ad_token=None,
        azure_ad_token_provider=None,
    )


def test_azure_chatopenai_init_no_deployment_uses_model_attr(
    mock_azure_clients: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test AzureChatOpenAI uses model attribute if azure_deployment is None."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=None,
        model=TEST_AZURE_MODEL_NAME,
    )
    assert chat_model.model == TEST_AZURE_MODEL_NAME
    mock_azure_clients['sync_constructor'].assert_called_once()


def test_azure_chatopenai_init_no_deployment_no_model_uses_default_model(
    mock_azure_clients: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test AzureChatOpenAI uses default ChatOpenAI model if azure_deployment is None and model arg is NOT GIVEN."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=None,
        # model is NOT passed, so it should use ChatOpenAI's default
    )
    assert chat_model.model == 'gpt-3.5-turbo'
    mock_azure_clients['sync_constructor'].assert_called_once()


def test_azure_chatopenai_init_all_custom_params(mock_azure_clients: dict) -> None:
    """Test AzureChatOpenAI with all custom parameters including inherited ones."""
    custom_temp = 0.12
    custom_max_tokens = 123
    custom_max_completion = 100
    custom_retries = 3
    custom_timeout = 45

    chat_model = AzureChatOpenAI(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        temperature=custom_temp,
        max_tokens=custom_max_tokens,
        max_completion_tokens=custom_max_completion,
        max_retries=custom_retries,
        timeout=custom_timeout,
    )
    assert chat_model.model == 'gpt-3.5-turbo'
    assert chat_model.temperature == custom_temp
    assert chat_model.max_tokens == custom_max_tokens
    assert chat_model.max_completion_tokens == custom_max_completion
    assert chat_model.max_retries == custom_retries
    assert chat_model.timeout == custom_timeout

    expected_client_kwargs = {
        'api_key': TEST_AZURE_API_KEY,
        'azure_endpoint': TEST_AZURE_ENDPOINT,
        'api_version': TEST_AZURE_API_VERSION,
        'max_retries': custom_retries,
        'timeout': custom_timeout,
        'azure_ad_token': None,
        'azure_ad_token_provider': None,
    }
    mock_azure_clients['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_azure_clients['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_azure_chatopenai_init_missing_endpoint_or_version_might_fail_at_client_sdk(
    mock_azure_clients: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that client SDK likely fails if endpoint/version are bad (e.g. empty/None)."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    expected_error = openai.APIError('Simulated SDK init error', request=MagicMock(), body=None)

    # Test with empty endpoint
    mock_azure_clients['sync_constructor'].side_effect = expected_error
    mock_azure_clients['async_constructor'].side_effect = expected_error
    with pytest.raises(openai.APIError, match='Simulated SDK init error'):
        AzureChatOpenAI(
            azure_endpoint='', api_version=TEST_AZURE_API_VERSION, azure_deployment=TEST_AZURE_DEPLOYMENT_NAME
        )
    mock_azure_clients['sync_constructor'].reset_mock()
    mock_azure_clients['async_constructor'].reset_mock()
    mock_azure_clients['sync_constructor'].side_effect = None
    mock_azure_clients['async_constructor'].side_effect = None

    # Test with None api_version
    mock_azure_clients['sync_constructor'].side_effect = expected_error
    mock_azure_clients['async_constructor'].side_effect = expected_error
    with pytest.raises(openai.APIError, match='Simulated SDK init error'):
        AzureChatOpenAI(
            azure_endpoint=TEST_AZURE_ENDPOINT, api_version=None, azure_deployment=TEST_AZURE_DEPLOYMENT_NAME
        )


# --- AzureChatOpenAI Invoke/Generate Tests (Model Resolution) --- #


@pytest.mark.parametrize('is_async', [False, True])
def test_azure_chatopenai_invoke_uses_correct_model_and_client(
    mock_azure_clients: dict,
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
    is_async: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invoke/ainvoke on AzureChatOpenAI uses Azure client and correct model (from deployment)."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    chat_model = AzureChatOpenAI(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        temperature=0.77,
    )
    assert chat_model.model == 'gpt-3.5-turbo'

    expected_api_kwargs = {
        'model': chat_model.model,
        'messages': sample_messages_dict_openai_fmt,
        'temperature': 0.77,
        'max_tokens': NOT_GIVEN,
        'max_completion_tokens': NOT_GIVEN,
    }

    if is_async:
        result = asyncio.run(chat_model.ainvoke(sample_messages))
        assert result.content == 'Mocked Azure async response'
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
        mock_azure_clients['sync_create'].assert_not_called()
    else:
        result = chat_model.invoke(sample_messages)
        assert result.content == 'Mocked Azure sync response'
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)
        mock_azure_clients['async_create'].assert_not_awaited()


@pytest.mark.parametrize('is_async', [False, True])
def test_azure_chatopenai_invoke_no_deployment_uses_model_attr(
    mock_azure_clients: dict,
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
    is_async: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invoke on AzureChatOpenAI uses model attribute if no deployment."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    chat_model = AzureChatOpenAI(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=None,
        model=TEST_AZURE_MODEL_NAME,
        temperature=0.78,
    )
    assert chat_model.model == TEST_AZURE_MODEL_NAME

    expected_api_kwargs = {
        'model': TEST_AZURE_MODEL_NAME,
        'messages': sample_messages_dict_openai_fmt,
        'temperature': 0.78,
        'max_tokens': NOT_GIVEN,
        'max_completion_tokens': NOT_GIVEN,
    }
    if is_async:
        result = asyncio.run(chat_model.ainvoke(sample_messages))
        assert result.content == 'Mocked Azure async response'
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
    else:
        result = chat_model.invoke(sample_messages)
        assert result.content == 'Mocked Azure sync response'
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)


@pytest.mark.parametrize('is_async', [False, True])
def test_azure_chatopenai_invoke_no_deployment_no_model_uses_default(
    mock_azure_clients: dict,
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
    is_async: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invoke on AzureChatOpenAI uses default model if no deployment/model."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    chat_model = AzureChatOpenAI(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        azure_deployment=None,
        # model is NOT passed here
        temperature=0.79,
    )
    assert chat_model.model == 'gpt-3.5-turbo'

    expected_api_kwargs = {
        'model': 'gpt-3.5-turbo',
        'messages': sample_messages_dict_openai_fmt,
        'temperature': 0.79,
        'max_tokens': NOT_GIVEN,
        'max_completion_tokens': NOT_GIVEN,
    }
    if is_async:
        result = asyncio.run(chat_model.ainvoke(sample_messages))
        assert result.content == 'Mocked Azure async response'
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
    else:
        result = chat_model.invoke(sample_messages)
        assert result.content == 'Mocked Azure sync response'
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)


# --- Tests for standalone from_client functions --- #


def test_chatopenai_from_client(mock_openai_clients: dict[str, Any]) -> None:
    from tinylcel.providers.openai.chat_models import from_client

    instance = from_client(
        client=mock_openai_clients['sync_client'],
        model='gpt-3.5-turbo',
        temperature=0.7,
        max_tokens=100,
        max_completion_tokens=50,
        max_retries=3,
        timeout=30,
    )
    assert isinstance(instance, ChatOpenAI)
    assert instance.model == 'gpt-3.5-turbo'
    assert instance.temperature == 0.7
    assert instance.max_tokens == 100
    assert instance.max_completion_tokens == 50
    assert instance.max_retries == 3
    assert instance.timeout == 30


# --- Tests for Azure standalone from_azure_client function --- #


def test_azure_chatopenai_from_client_success(
    mock_azure_clients: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tinylcel.providers.openai.chat_models import from_azure_client

    # Set up the required environment variable for the test
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)

    instance = from_azure_client(
        client=mock_azure_clients['sync_client'],
        model=TEST_AZURE_MODEL_NAME,
        temperature=0.7,
        max_tokens=100,
        max_completion_tokens=50,
        max_retries=3,
        timeout=30,
    )
    assert isinstance(instance, AzureChatOpenAI)
    assert instance.model == TEST_AZURE_MODEL_NAME
    assert instance.temperature == 0.7
    assert instance.max_tokens == 100
    assert instance.max_completion_tokens == 50
    assert instance.max_retries == 3
    assert instance.timeout == 30


# --- Tests for Structured Output --- #


@pytest.mark.usefixtures('mock_openai_clients')
def test_base_chat_model_with_structured_output_returns_sequence() -> None:
    """Test BaseChatModel.with_structured_output is abstract - subclasses must implement it."""
    # Test that the base implementation is abstract by checking that ChatOpenAI overrides it
    model = ChatOpenAI()

    # ChatOpenAI should override the abstract method and return StructureChatOpenAI
    structured_model = model.with_structured_output(Person)
    assert isinstance(structured_model, StructureChatOpenAI)
    assert structured_model.output_type is Person


@pytest.mark.usefixtures('mock_openai_clients')
def test_chatopenai_with_structured_output_returns_structured_model() -> None:
    """Test ChatOpenAI.with_structured_output returns StructureChatOpenAI instance."""
    model = ChatOpenAI(api_key='test-key', model='gpt-4o-mini', temperature=0.7, max_tokens=100)
    structured_model = model.with_structured_output(Person)

    # Should return StructureChatOpenAI
    assert isinstance(structured_model, StructureChatOpenAI)

    # Should inherit all settings from parent model
    assert structured_model.model == 'gpt-4o-mini'
    assert structured_model.api_key == 'test-key'
    assert structured_model.temperature == 0.7
    assert structured_model.max_tokens == 100
    assert structured_model.output_type is Person


@pytest.mark.usefixtures('mock_openai_clients')
def test_chatopenai_with_structured_output_copies_all_settings() -> None:
    """Test that with_structured_output copies all settings from parent model."""
    model = ChatOpenAI(
        api_key='test-key',
        model='gpt-4',
        temperature=0.2,
        max_tokens=50,
        max_completion_tokens=25,
        max_retries=5,
        timeout=30,
        base_url='http://localhost:1234',
    )
    structured_model = model.with_structured_output(SimpleResponse)

    assert structured_model.api_key == 'test-key'
    assert structured_model.model == 'gpt-4'
    assert structured_model.temperature == 0.2
    assert structured_model.max_tokens == 50
    assert structured_model.max_completion_tokens == 25
    assert structured_model.max_retries == 5
    assert structured_model.timeout == 30
    assert structured_model.base_url == 'http://localhost:1234'
    assert structured_model.output_type is SimpleResponse


def test_structured_chat_openai_generate_success(
    mock_structured_output_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    """Test StructureChatOpenAI._generate calls beta.chat.completions.parse and returns parsed model."""
    # Set up parsed response
    test_person = Person(name='John', age=25, city='NYC')
    mock_structured_output_clients['sync_parse_response'].choices[0].message.parsed = test_person

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)
    result = structured_model._generate(sample_messages)

    # Should return the parsed Pydantic model directly
    assert result is test_person
    assert isinstance(result, Person)
    assert result.name == 'John'
    assert result.age == 25
    assert result.city == 'NYC'

    # Should call beta.chat.completions.parse with correct parameters
    mock_structured_output_clients['sync_parse'].assert_called_once_with(
        model='gpt-4o-mini',
        messages=sample_messages_dict_openai_fmt,
        response_format=Person,
        temperature=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN,
    )


@pytest.mark.asyncio
async def test_structured_chat_openai_agenerate_success(
    mock_structured_output_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    """Test StructureChatOpenAI._agenerate calls async beta.chat.completions.parse and returns parsed model."""
    # Set up parsed response
    test_response = SimpleResponse(message='Success', success=True)
    mock_structured_output_clients['async_parse_response'].choices[0].message.parsed = test_response

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key', temperature=0.5)
    structured_model: StructureChatOpenAI[SimpleResponse] = base_model.with_structured_output(SimpleResponse)
    result = await structured_model._agenerate(sample_messages)

    # Should return the parsed Pydantic model directly
    assert result is test_response
    assert isinstance(result, SimpleResponse)
    assert result.message == 'Success'
    assert result.success is True

    # Should call async beta.chat.completions.parse with correct parameters
    mock_structured_output_clients['async_parse'].assert_awaited_once_with(
        model='gpt-4o-mini',
        messages=sample_messages_dict_openai_fmt,
        response_format=SimpleResponse,
        temperature=0.5,
        max_completion_tokens=NOT_GIVEN,
    )


def test_structured_chat_openai_generate_with_all_params(
    mock_structured_output_clients: dict[str, Any],
    sample_messages: MessagesInput,
    sample_messages_dict_openai_fmt: list[dict[str, str]],
) -> None:
    """Test StructureChatOpenAI._generate passes all parameters correctly."""
    test_person = Person(name='Jane', age=30, city='SF')
    mock_structured_output_clients['sync_parse_response'].choices[0].message.parsed = test_person

    base_model = ChatOpenAI(
        model='gpt-4',
        api_key='test-key',
        temperature=0.2,
        max_tokens=100,
        max_completion_tokens=50,
    )
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)
    result = structured_model._generate(sample_messages)

    assert result is test_person

    mock_structured_output_clients['sync_parse'].assert_called_once_with(
        model='gpt-4',
        messages=sample_messages_dict_openai_fmt,
        response_format=Person,
        temperature=0.2,
        max_completion_tokens=50,
    )


def test_structured_chat_openai_generate_refusal_error(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI._generate raises ValueError on parsing failure."""
    # Set up parsing failure response
    mock_structured_output_clients['sync_parse_response'].choices[0].message.parsed = None

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)

    with pytest.raises(ValueError, match='OpenAI error: Parsing failed'):
        structured_model._generate(sample_messages)


@pytest.mark.asyncio
async def test_structured_chat_openai_agenerate_refusal_error(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI._agenerate raises ValueError on parsing failure."""
    # Set up parsing failure response
    mock_structured_output_clients['async_parse_response'].choices[0].message.parsed = None

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[SimpleResponse] = base_model.with_structured_output(SimpleResponse)

    with pytest.raises(ValueError, match='OpenAI error: Parsing failed'):
        await structured_model._agenerate(sample_messages)


def test_structured_chat_openai_generate_parsing_failure(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI._generate raises ValueError when output_parsed is None."""
    # Set up response with None
    mock_structured_output_clients['sync_parse_response'].choices[0].message.parsed = None

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)

    with pytest.raises(ValueError, match='OpenAI error: Parsing failed'):
        structured_model._generate(sample_messages)


def test_structured_chat_openai_generate_api_error(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI._generate propagates OpenAI API errors."""
    mock_structured_output_clients['sync_parse'].side_effect = openai.APIError('API Failed', request=Mock(), body=None)

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)

    with pytest.raises(openai.APIError, match='API Failed'):
        structured_model._generate(sample_messages)


@pytest.mark.asyncio
async def test_structured_chat_openai_agenerate_api_error(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI._agenerate propagates OpenAI API errors."""
    mock_structured_output_clients['async_parse'].side_effect = openai.APIError(
        'Async API Failed', request=Mock(), body=None
    )

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[SimpleResponse] = base_model.with_structured_output(SimpleResponse)

    with pytest.raises(openai.APIError, match='Async API Failed'):
        await structured_model._agenerate(sample_messages)


def test_structured_chat_openai_invoke_success(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI.invoke returns parsed model directly."""
    test_person = Person(name='Alice', age=28, city='Boston')
    mock_structured_output_clients['sync_parse_response'].choices[0].message.parsed = test_person

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)
    result = structured_model.invoke(sample_messages)

    # invoke should return the parsed model directly, not an AIMessage
    assert result is test_person


@pytest.mark.asyncio
async def test_structured_chat_openai_ainvoke_success(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI.ainvoke returns parsed model directly."""
    test_response = SimpleResponse(message='Done', success=True)
    mock_structured_output_clients['async_parse_response'].choices[0].message.parsed = test_response

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[SimpleResponse] = base_model.with_structured_output(SimpleResponse)
    result = await structured_model.ainvoke(sample_messages)

    # ainvoke should return the parsed model directly, not an AIMessage
    assert result is test_response


def test_structured_chat_openai_batch_success(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI.batch returns multiple parsed models."""
    test_person1 = Person(name='Bob', age=35, city='Seattle')
    test_person2 = Person(name='Carol', age=42, city='Austin')

    # Set up multiple responses
    mock_response1 = MagicMock()
    mock_response1.choices = [MagicMock()]
    mock_response1.choices[0].message = MagicMock()
    mock_response1.choices[0].message.parsed = test_person1

    mock_response2 = MagicMock()
    mock_response2.choices = [MagicMock()]
    mock_response2.choices[0].message = MagicMock()
    mock_response2.choices[0].message.parsed = test_person2

    mock_structured_output_clients['sync_parse'].side_effect = [
        mock_response1,
        mock_response2,
    ]

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[Person] = base_model.with_structured_output(Person)
    results = structured_model.batch([sample_messages, sample_messages])

    assert len(results) == 2
    assert results[0] is test_person1
    assert results[1] is test_person2


@pytest.mark.asyncio
async def test_structured_chat_openai_abatch_success(
    mock_structured_output_clients: dict[str, Any], sample_messages: MessagesInput
) -> None:
    """Test StructureChatOpenAI.abatch returns multiple parsed models."""
    test_response1 = SimpleResponse(message='First', success=True)
    test_response2 = SimpleResponse(message='Second', success=False)

    # Set up multiple async responses
    mock_response1 = MagicMock()
    mock_response1.choices = [MagicMock()]
    mock_response1.choices[0].message = MagicMock()
    mock_response1.choices[0].message.parsed = test_response1

    mock_response2 = MagicMock()
    mock_response2.choices = [MagicMock()]
    mock_response2.choices[0].message = MagicMock()
    mock_response2.choices[0].message.parsed = test_response2

    mock_structured_output_clients['async_parse'].side_effect = [
        mock_response1,
        mock_response2,
    ]

    base_model = ChatOpenAI(model='gpt-4o-mini', api_key='test-key')
    structured_model: StructureChatOpenAI[SimpleResponse] = base_model.with_structured_output(SimpleResponse)
    results = await structured_model.abatch([sample_messages, sample_messages])

    assert len(results) == 2
    assert results[0] is test_response1
    assert results[1] is test_response2
