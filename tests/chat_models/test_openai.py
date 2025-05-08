"""Tests for the ChatOpenAI chat model implementation."""

import asyncio
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import openai
import pytest
from openai import NOT_GIVEN
from openai import DEFAULT_MAX_RETRIES

from tinylcel.messages import AIMessage
from tinylcel.messages import MessagesInput
from tinylcel.messages import HumanMessage
from tinylcel.messages import SystemMessage
from tinylcel.chat_models.openai import ChatOpenAI
from tinylcel.chat_models.openai import _get_openai_api_key
from tinylcel.chat_models.openai import AzureChatOpenAI


# Constants for Azure tests
TEST_AZURE_API_KEY = "test-azure-api-key"
TEST_AZURE_ENDPOINT = "https://test-azure-endpoint.openai.azure.com"
TEST_AZURE_API_VERSION = "2023-07-01-preview"
TEST_AZURE_DEPLOYMENT_NAME = "test-deployment"
TEST_AZURE_MODEL_NAME = "gpt-35-turbo-test"


@pytest.fixture
def mock_openai_clients():
    """Fixture to mock openai.OpenAI and openai.AsyncOpenAI clients."""
    mock_sync_client = MagicMock(spec=openai.OpenAI)
    mock_sync_completions = MagicMock()
    mock_sync_chat = MagicMock()
    mock_sync_client.chat = mock_sync_chat
    mock_sync_chat.completions = mock_sync_completions
    mock_sync_response = MagicMock()
    mock_sync_choice = MagicMock()
    mock_sync_message = MagicMock()
    mock_sync_message.content = "Mocked sync response"
    mock_sync_choice.message = mock_sync_message
    mock_sync_response.choices = [mock_sync_choice]
    mock_sync_completions.create.return_value = mock_sync_response

    mock_async_response = MagicMock()
    mock_async_choice = MagicMock()
    mock_async_message = MagicMock()
    mock_async_message.content = "Mocked async response"
    mock_async_choice.message = mock_async_message
    mock_async_response.choices = [mock_async_choice]
    mock_async_create_method = AsyncMock(return_value=mock_async_response)

    with patch('tinylcel.chat_models.openai.openai.OpenAI', return_value=mock_sync_client) as mock_sync_constructor:
        with patch('tinylcel.chat_models.openai.openai.AsyncOpenAI') as mock_async_constructor_patch:
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
                'async_constructor': mock_async_constructor_patch
            }

@pytest.fixture
def mock_azure_clients():
    """Fixture to mock openai.AzureOpenAI and openai.AsyncAzureOpenAI clients."""
    mock_sync_azure_client = MagicMock(spec=openai.AzureOpenAI)
    mock_sync_azure_completions = MagicMock()
    mock_sync_azure_chat = MagicMock()
    mock_sync_azure_client.chat = mock_sync_azure_chat
    mock_sync_azure_chat.completions = mock_sync_azure_completions
    mock_sync_azure_response = MagicMock()
    mock_sync_azure_choice = MagicMock()
    mock_sync_azure_message = MagicMock()
    mock_sync_azure_message.content = "Mocked Azure sync response"
    mock_sync_azure_choice.message = mock_sync_azure_message
    mock_sync_azure_response.choices = [mock_sync_azure_choice]
    mock_sync_azure_completions.create.return_value = mock_sync_azure_response

    mock_async_azure_response = MagicMock()
    mock_async_azure_choice = MagicMock()
    mock_async_azure_message = MagicMock()
    mock_async_azure_message.content = "Mocked Azure async response"
    mock_async_azure_choice.message = mock_async_azure_message
    mock_async_azure_response.choices = [mock_async_azure_choice]
    mock_async_azure_create_method = AsyncMock(return_value=mock_async_azure_response)

    with patch('tinylcel.chat_models.openai.AzureOpenAI', return_value=mock_sync_azure_client) as mock_sync_azure_constructor:
        with patch('tinylcel.chat_models.openai.AsyncAzureOpenAI') as mock_async_azure_constructor_patch:
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
                'async_constructor': mock_async_azure_constructor_patch
            }

@pytest.fixture
def sample_messages() -> MessagesInput:
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]

@pytest.fixture
def sample_messages_dict_openai_fmt() -> list[dict[str, str]]:
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the capital of France?'},
    ]

@pytest.fixture(autouse=True)
def set_openai_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-env")

# --- Test Cases --- #

def test_get_openai_api_key_from_arg():
    assert _get_openai_api_key("test-key-arg") == "test-key-arg"

def test_get_openai_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-env")
    assert _get_openai_api_key(None) == "test-key-env"

def test_get_openai_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
        _get_openai_api_key(None)

def test_chatopenai_initialization_defaults(mock_openai_clients):
    chat_model = ChatOpenAI()
    assert chat_model.model == 'gpt-3.5-turbo'
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_tokens is NOT_GIVEN
    assert chat_model.max_completion_tokens is NOT_GIVEN
    assert chat_model.openai_api_key is None
    assert chat_model.max_retries == DEFAULT_MAX_RETRIES
    # timeout field in ChatOpenAI now defaults to NOT_GIVEN
    assert chat_model.timeout is NOT_GIVEN

    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key="test-key-from-env",
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN # ChatOpenAI passes its self.timeout (which is NOT_GIVEN by default)
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key="test-key-from-env",
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN # ChatOpenAI passes its self.timeout
    )

def test_chatopenai_initialization_explicit_key(mock_openai_clients):
    chat_model = ChatOpenAI(openai_api_key="explicit-test-key")
    assert chat_model.openai_api_key == "explicit-test-key"
    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key="explicit-test-key",
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN # Default timeout for ChatOpenAI is NOT_GIVEN
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key="explicit-test-key",
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN # Default timeout for ChatOpenAI is NOT_GIVEN
    )

def test_chatopenai_initialization_custom_params(mock_openai_clients):
    # ChatOpenAI timeout field is now int | NotGiven
    custom_timeout_int = 30 
    chat_model = ChatOpenAI(
        model='gpt-4',
        temperature=0.1,
        max_tokens=150,
        max_completion_tokens=100,
        openai_api_key='custom-key',
        max_retries=5,
        timeout=custom_timeout_int
    )
    assert chat_model.model == 'gpt-4'
    assert chat_model.temperature == 0.1
    assert chat_model.max_tokens == 150
    assert chat_model.max_completion_tokens == 100
    assert chat_model.max_retries == 5
    assert chat_model.timeout == custom_timeout_int
    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key="custom-key",
        max_retries=5,
        timeout=custom_timeout_int # The int value is passed to the client
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key="custom-key",
        max_retries=5,
        timeout=custom_timeout_int # The int value is passed to the client
    )

def test_chatopenai_initialization_with_max_tokens_only(mock_openai_clients):
    chat_model = ChatOpenAI(max_tokens=100, openai_api_key='mt-key')
    assert chat_model.max_tokens == 100
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_completion_tokens is NOT_GIVEN
    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key="mt-key", max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key="mt-key", max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )

def test_chatopenai_initialization_with_max_completion_tokens_only(mock_openai_clients):
    chat_model = ChatOpenAI(max_completion_tokens=200, openai_api_key='mct-key')
    assert chat_model.max_completion_tokens == 200
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_tokens is NOT_GIVEN
    mock_openai_clients['sync_constructor'].assert_called_once_with(
        api_key="mct-key", max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_clients['async_constructor'].assert_called_once_with(
        api_key="mct-key", max_retries=DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )

@pytest.mark.parametrize(
    "input_message, expected_dict",
    [
        (SystemMessage(content="System prompt"), {'role': 'system', 'content': 'System prompt'}),
        (HumanMessage(content="User query"), {'role': 'user', 'content': 'User query'}),
        (AIMessage(content="Assistant response"), {'role': 'assistant', 'content': 'Assistant response'}),
    ]
)
def test_convert_message_to_dict(mock_openai_clients, input_message, expected_dict):
    chat_model = ChatOpenAI()
    assert chat_model._convert_message_to_dict(input_message) == expected_dict

# --- Invoke / _generate Tests --- #

def test_chatopenai_generate_success_defaults(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI()
    result = chat_model._generate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mocked sync response"
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN
    )

@pytest.mark.asyncio
async def test_chatopenai_agenerate_success_defaults(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI()
    result = await chat_model._agenerate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mocked async response"
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN
    )

def test_chatopenai_generate_with_all_optional_params_set(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI(temperature=0.2, max_tokens=50, max_completion_tokens=25)
    result = chat_model._generate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mocked sync response"
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=0.2,
        max_tokens=50,
        max_completion_tokens=25
    )

@pytest.mark.asyncio
async def test_chatopenai_agenerate_with_all_optional_params_set(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI(temperature=0.3, max_tokens=60, max_completion_tokens=30)
    result = await chat_model._agenerate(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mocked async response"
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=0.3,
        max_tokens=60,
        max_completion_tokens=30
    )

def test_chatopenai_generate_only_max_tokens_set(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI(max_tokens=50)
    chat_model._generate(sample_messages)
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=50,
        max_completion_tokens=NOT_GIVEN
    )

@pytest.mark.asyncio
async def test_chatopenai_agenerate_only_max_tokens_set(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI(max_tokens=75)
    await chat_model._agenerate(sample_messages)
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=75,
        max_completion_tokens=NOT_GIVEN
    )

def test_chatopenai_invoke_success_defaults(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI()
    result = chat_model.invoke(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mocked sync response"
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN
    )

@pytest.mark.asyncio
async def test_chatopenai_ainvoke_success_defaults(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    chat_model = ChatOpenAI()
    result = await chat_model.ainvoke(sample_messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mocked async response"
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        max_completion_tokens=NOT_GIVEN
    )

def test_chatopenai_generate_api_error(mock_openai_clients, sample_messages):
    mock_openai_clients['sync_create'].side_effect = openai.APIError("API Failed", request=Mock(), body=None)
    chat_model = ChatOpenAI()
    with pytest.raises(openai.APIError, match="API Failed"):
        chat_model._generate(sample_messages)

def test_chatopenai_generate_none_content(mock_openai_clients, sample_messages):
    mock_response = mock_openai_clients['sync_create'].return_value
    mock_response.choices[0].message.content = None
    chat_model = ChatOpenAI()
    with pytest.raises(ValueError, match="OpenAI response content is None"):
        chat_model._generate(sample_messages)

@pytest.mark.asyncio
async def test_chatopenai_agenerate_api_error(mock_openai_clients, sample_messages):
    mock_openai_clients['async_create'].side_effect = openai.APIError("Async API Failed", request=Mock(), body=None)
    chat_model = ChatOpenAI()
    with pytest.raises(openai.APIError, match="Async API Failed"):
        await chat_model._agenerate(sample_messages)

@pytest.mark.asyncio
async def test_chatopenai_agenerate_none_content(mock_openai_clients, sample_messages):
    mock_response = mock_openai_clients['async_create'].return_value
    mock_response.choices[0].message.content = None
    mock_openai_clients['async_create'].side_effect = None 
    chat_model = ChatOpenAI()
    with pytest.raises(ValueError, match="OpenAI response content is None"):
        await chat_model._agenerate(sample_messages)

# --- Batch Tests --- #
def test_chatopenai_batch_success(mock_openai_clients, sample_messages):
    chat_model = ChatOpenAI()
    inputs = [sample_messages, sample_messages] 
    mock_openai_clients['sync_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 2"))]),
    ]
    results = chat_model.batch(inputs)
    assert len(results) == 2
    assert results[0].content == "Sync Resp 1"
    assert results[1].content == "Sync Resp 2"
    assert mock_openai_clients['sync_create'].call_count == 2

def test_chatopenai_batch_return_exceptions(mock_openai_clients, sample_messages):
    chat_model = ChatOpenAI()
    inputs = [sample_messages, [HumanMessage(content="error trigger")], sample_messages]
    error = ValueError("Batch Error")
    mock_openai_clients['sync_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 1"))]),
        error,
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 3"))]),
    ]
    results = chat_model.batch(inputs, return_exceptions=True)
    assert len(results) == 3
    assert results[0].content == "Sync Resp 1"
    assert results[1] is error
    assert results[2].content == "Sync Resp 3"
    assert mock_openai_clients['sync_create'].call_count == 3

@pytest.mark.asyncio
async def test_chatopenai_abatch_success(mock_openai_clients, sample_messages):
    chat_model = ChatOpenAI()
    inputs = [sample_messages, sample_messages]
    mock_openai_clients['async_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 2"))]),
    ]
    results = await chat_model.abatch(inputs)
    assert len(results) == 2
    assert results[0].content == "Async Resp 1"
    assert results[1].content == "Async Resp 2"
    assert mock_openai_clients['async_create'].await_count == 2

@pytest.mark.asyncio
async def test_chatopenai_abatch_return_exceptions(mock_openai_clients, sample_messages):
    chat_model = ChatOpenAI()
    inputs = [sample_messages, [HumanMessage(content="error trigger")], sample_messages]
    error = ValueError("Async Batch Error")
    mock_openai_clients['async_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 1"))]),
        error,
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 3"))]),
    ]
    results = await chat_model.abatch(inputs, return_exceptions=True)
    assert len(results) == 3
    assert results[0].content == "Async Resp 1"
    assert results[1] is error
    assert results[2].content == "Async Resp 3"
    assert mock_openai_clients['async_create'].await_count == 3

# --- AzureChatOpenAI Tests --- #

def test_azure_chatopenai_initialization_defaults_env_key(mock_azure_clients, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", TEST_AZURE_API_KEY)
    # Pass an explicit timeout to AzureChatOpenAI to distinguish from ChatOpenAI's default issues
    # This also ensures the Azure client constructor is tested with a specific Timeout object.
    # With ChatOpenAI.timeout now being int | NotGiven, this test needs adjustment.
    # Let's test passing an int for timeout.
    explicit_timeout_int_for_azure = 50
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        timeout=explicit_timeout_int_for_azure 
    )
    assert chat_model.azure_endpoint == TEST_AZURE_ENDPOINT
    assert chat_model.api_version == TEST_AZURE_API_VERSION
    assert chat_model.openai_api_key is None
    assert chat_model.azure_deployment is None
    assert chat_model.model == 'gpt-3.5-turbo'
    assert chat_model.temperature is NOT_GIVEN
    assert chat_model.max_tokens is NOT_GIVEN
    assert chat_model.max_completion_tokens is NOT_GIVEN
    assert chat_model.max_retries == DEFAULT_MAX_RETRIES 
    assert chat_model.timeout == explicit_timeout_int_for_azure

    mock_azure_clients['sync_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=explicit_timeout_int_for_azure
    )
    mock_azure_clients['async_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=explicit_timeout_int_for_azure
    )

def test_azure_chatopenai_initialization_explicit_key_and_deployment(mock_azure_clients):
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME
    )
    assert chat_model.azure_deployment == TEST_AZURE_DEPLOYMENT_NAME
    assert chat_model.model == TEST_AZURE_DEPLOYMENT_NAME
    assert chat_model.openai_api_key == TEST_AZURE_API_KEY
    # Assert that inherited defaults for timeout/retries are passed if not overridden
    assert chat_model.timeout is NOT_GIVEN # Inherited from ChatOpenAI field default (NOT_GIVEN)
    mock_azure_clients['sync_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN # Expecting ChatOpenAI's default (NOT_GIVEN) to be used
    )
    mock_azure_clients['async_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=DEFAULT_MAX_RETRIES,
        timeout=NOT_GIVEN # Expecting ChatOpenAI's default (NOT_GIVEN) to be used
    )

def test_azure_chatopenai_initialization_custom_all_params(mock_azure_clients):
    custom_azure_timeout_int = 45
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        temperature=0.9,
        max_tokens=250,
        max_completion_tokens=120,
        max_retries=7,
        timeout=custom_azure_timeout_int
    )
    assert chat_model.model == TEST_AZURE_DEPLOYMENT_NAME
    assert chat_model.temperature == 0.9
    assert chat_model.max_tokens == 250
    assert chat_model.max_completion_tokens == 120
    assert chat_model.max_retries == 7
    assert chat_model.timeout == custom_azure_timeout_int
    mock_azure_clients['sync_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=7,
        timeout=custom_azure_timeout_int
    )
    mock_azure_clients['async_constructor'].assert_called_once_with(
        api_key=TEST_AZURE_API_KEY,
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        max_retries=7,
        timeout=custom_azure_timeout_int
    )

def test_azure_chatopenai_initialization_deployment_overrides_model(mock_azure_clients):
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        model="some-other-model-to-be-overridden"
    )
    assert chat_model.model == TEST_AZURE_DEPLOYMENT_NAME

# --- Azure _generate / _agenerate Tests --- #

@pytest.mark.parametrize("is_async", [False, True])
def test_azure_chatopenai_generate_with_deployment_defaults(
    mock_azure_clients, sample_messages, sample_messages_dict_openai_fmt, is_async
):
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
    )
    assert chat_model.model == TEST_AZURE_DEPLOYMENT_NAME
    expected_api_kwargs = {
        "model": TEST_AZURE_DEPLOYMENT_NAME,
        "messages": sample_messages_dict_openai_fmt,
        "temperature": NOT_GIVEN,
        "max_tokens": NOT_GIVEN,
        "max_completion_tokens": NOT_GIVEN,
    }
    if is_async:
        result = asyncio.run(chat_model._agenerate(sample_messages))
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
    else:
        result = chat_model._generate(sample_messages)
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)
    assert isinstance(result, AIMessage)

@pytest.mark.parametrize("is_async", [False, True])
def test_azure_chatopenai_generate_with_all_optional_params_set(
    mock_azure_clients, sample_messages, sample_messages_dict_openai_fmt, is_async
):
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        temperature=0.5,
        max_tokens=150,
        max_completion_tokens=70
    )
    assert chat_model.model == TEST_AZURE_DEPLOYMENT_NAME
    expected_api_kwargs = {
        "model": TEST_AZURE_DEPLOYMENT_NAME,
        "messages": sample_messages_dict_openai_fmt,
        "temperature": 0.5,
        "max_tokens": 150,
        "max_completion_tokens": 70
    }
    if is_async:
        result = asyncio.run(chat_model._agenerate(sample_messages))
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
    else:
        result = chat_model._generate(sample_messages)
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)
    assert isinstance(result, AIMessage)

@pytest.mark.parametrize("is_async", [False, True])
def test_azure_chatopenai_generate_without_deployment_uses_model_attr(
    mock_azure_clients, sample_messages, sample_messages_dict_openai_fmt, is_async
):
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        model=TEST_AZURE_MODEL_NAME, 
        temperature=0.6
    )
    assert chat_model.model == TEST_AZURE_MODEL_NAME
    expected_api_kwargs = {
        "model": TEST_AZURE_MODEL_NAME,
        "messages": sample_messages_dict_openai_fmt,
        "temperature": 0.6,
        "max_tokens": NOT_GIVEN,
        "max_completion_tokens": NOT_GIVEN
    }
    if is_async:
        result = asyncio.run(chat_model._agenerate(sample_messages)) # Assign result
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
    else:
        result = chat_model._generate(sample_messages) # Assign result
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)
    assert isinstance(result, AIMessage)

@pytest.mark.parametrize("is_async", [False, True])
def test_azure_chatopenai_generate_with_only_max_tokens(
    mock_azure_clients, sample_messages, sample_messages_dict_openai_fmt, is_async
):
    chat_model = AzureChatOpenAI(
        azure_endpoint=TEST_AZURE_ENDPOINT,
        api_version=TEST_AZURE_API_VERSION,
        openai_api_key=TEST_AZURE_API_KEY,
        azure_deployment=TEST_AZURE_DEPLOYMENT_NAME,
        max_tokens=150
    )
    assert chat_model.model == TEST_AZURE_DEPLOYMENT_NAME
    expected_api_kwargs = {
        "model": TEST_AZURE_DEPLOYMENT_NAME,
        "messages": sample_messages_dict_openai_fmt,
        "temperature": NOT_GIVEN, 
        "max_tokens": 150,
        "max_completion_tokens": NOT_GIVEN
    }
    if is_async:
        result = asyncio.run(chat_model._agenerate(sample_messages))
        mock_azure_clients['async_create'].assert_awaited_once_with(**expected_api_kwargs)
    else:
        result = chat_model._generate(sample_messages)
        mock_azure_clients['sync_create'].assert_called_once_with(**expected_api_kwargs)
    assert isinstance(result, AIMessage)

def test_azure_chatopenai_generate_api_error(mock_azure_clients, sample_messages):
    mock_azure_clients['sync_create'].side_effect = openai.APIError("Azure API Failed", request=Mock(), body=None)
    chat_model = AzureChatOpenAI(azure_endpoint=TEST_AZURE_ENDPOINT, api_version=TEST_AZURE_API_VERSION, openai_api_key=TEST_AZURE_API_KEY)
    with pytest.raises(openai.APIError, match="Azure API Failed"):
        chat_model._generate(sample_messages)

@pytest.mark.asyncio
async def test_azure_chatopenai_agenerate_api_error(mock_azure_clients, sample_messages):
    mock_azure_clients['async_create'].side_effect = openai.APIError("Azure Async API Failed", request=Mock(), body=None)
    chat_model = AzureChatOpenAI(azure_endpoint=TEST_AZURE_ENDPOINT, api_version=TEST_AZURE_API_VERSION, openai_api_key=TEST_AZURE_API_KEY)
    with pytest.raises(openai.APIError, match="Azure Async API Failed"):
        await chat_model._agenerate(sample_messages)

def test_azure_chatopenai_generate_none_content(mock_azure_clients, sample_messages):
    mock_response = mock_azure_clients['sync_create'].return_value
    mock_response.choices[0].message.content = None
    chat_model = AzureChatOpenAI(azure_endpoint=TEST_AZURE_ENDPOINT, api_version=TEST_AZURE_API_VERSION, openai_api_key=TEST_AZURE_API_KEY)
    with pytest.raises(ValueError, match="OpenAI response content is None"):
        chat_model._generate(sample_messages)

@pytest.mark.asyncio
async def test_azure_chatopenai_agenerate_none_content(mock_azure_clients, sample_messages):
    inner_mock_response = mock_azure_clients['async_create'].return_value
    inner_mock_response.choices[0].message.content = None
    mock_azure_clients['async_create'].side_effect = None
    chat_model = AzureChatOpenAI(azure_endpoint=TEST_AZURE_ENDPOINT, api_version=TEST_AZURE_API_VERSION, openai_api_key=TEST_AZURE_API_KEY)
    with pytest.raises(ValueError, match="OpenAI response content is None"):
        await chat_model._agenerate(sample_messages) 