"""Tests for the ChatOpenAI chat model implementation."""

import os
import pytest
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

import openai

from tinylcel.messages import AIMessage
from tinylcel.messages import HumanMessage
from tinylcel.messages import SystemMessage
from tinylcel.messages import MessagesInput
from tinylcel.chat_models.openai import ChatOpenAI
from tinylcel.chat_models.openai import _get_openai_api_key

# --- Fixtures --- #

@pytest.fixture
def mock_openai_clients():
    """Fixture to mock openai.OpenAI and openai.AsyncOpenAI clients."""
    # Mock the synchronous client and its nested structure
    mock_sync_client = MagicMock(spec=openai.OpenAI)
    mock_sync_completions = MagicMock()
    mock_sync_chat = MagicMock()
    mock_sync_client.chat = mock_sync_chat
    mock_sync_chat.completions = mock_sync_completions

    # Mock the response structure for sync
    mock_sync_response = MagicMock()
    mock_sync_choice = MagicMock()
    mock_sync_message = MagicMock()
    mock_sync_message.content = "Mocked sync response"
    mock_sync_choice.message = mock_sync_message
    mock_sync_response.choices = [mock_sync_choice]
    mock_sync_completions.create.return_value = mock_sync_response

    # Mock the asynchronous response structure
    mock_async_response = MagicMock()
    mock_async_choice = MagicMock()
    mock_async_message = MagicMock()
    mock_async_message.content = "Mocked async response"
    mock_async_choice.message = mock_async_message
    mock_async_response.choices = [mock_async_choice]
    # The async method itself
    mock_async_create_method = AsyncMock(return_value=mock_async_response)

    # Patch the client instantiation in the openai module scope using nested contexts
    with patch('tinylcel.chat_models.openai.openai.OpenAI', return_value=mock_sync_client) as mock_sync:
        with patch('tinylcel.chat_models.openai.openai.AsyncOpenAI') as mock_async_constructor:

            # Configure the instance that the AsyncOpenAI constructor will return
            # including the nested structure
            mock_async_client_instance = MagicMock()
            mock_async_completions = MagicMock()
            mock_async_chat = MagicMock()
            mock_async_client_instance.chat = mock_async_chat
            mock_async_chat.completions = mock_async_completions
            mock_async_completions.create = mock_async_create_method # Assign the AsyncMock here
            mock_async_constructor.return_value = mock_async_client_instance

            yield {
                'sync_client': mock_sync_client,
                'async_client': mock_async_client_instance,
                'sync_create': mock_sync_completions.create, # Point to the mock create method
                'async_create': mock_async_create_method,
                'sync_constructor': mock_sync,
                'async_constructor': mock_async_constructor
            }

@pytest.fixture
def sample_messages() -> MessagesInput:
    """Fixture for a sample list of messages."""
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]

@pytest.fixture
def sample_messages_dict_openai_fmt() -> list[dict[str, str]]:
    """OpenAI formatted version of sample_messages."""
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the capital of France?'},
    ]

# Make sure OPENAI_API_KEY is set for initialization tests if not mocked
@pytest.fixture(autouse=True)
def set_openai_key_env(monkeypatch):
    """Set dummy API key in environment for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-env")


# --- Test Cases --- #

def test_get_openai_api_key_from_arg():
    """Test API key retrieval prefers the argument."""
    assert _get_openai_api_key("test-key-arg") == "test-key-arg"

def test_get_openai_api_key_from_env(monkeypatch):
    """Test API key retrieval from environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-env")
    assert _get_openai_api_key(None) == "test-key-env"

def test_get_openai_api_key_missing(monkeypatch):
    """Test API key retrieval raises ValueError when missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
        _get_openai_api_key(None)

def test_chatopenai_initialization_defaults(mock_openai_clients):
    """Test ChatOpenAI initializes with default env key and clients."""
    chat_model = ChatOpenAI()
    assert chat_model.model == 'gpt-3.5-turbo'
    assert chat_model.temperature == 0.7
    assert chat_model.max_tokens is None # Check default max_tokens
    assert chat_model.openai_api_key is None # Default is None

    # Check constructors were called with the key from env
    mock_openai_clients['sync_constructor'].assert_called_once_with(api_key="test-key-from-env")
    mock_openai_clients['async_constructor'].assert_called_once_with(api_key="test-key-from-env")
    assert chat_model._client is mock_openai_clients['sync_client']
    assert chat_model._async_client is mock_openai_clients['async_client']

def test_chatopenai_initialization_explicit_key(mock_openai_clients):
    """Test ChatOpenAI initializes with an explicit API key."""
    chat_model = ChatOpenAI(openai_api_key="explicit-test-key")
    assert chat_model.openai_api_key == "explicit-test-key"

    # Check constructors were called with the explicit key
    mock_openai_clients['sync_constructor'].assert_called_once_with(api_key="explicit-test-key")
    mock_openai_clients['async_constructor'].assert_called_once_with(api_key="explicit-test-key")
    assert chat_model._client is mock_openai_clients['sync_client']
    assert chat_model._async_client is mock_openai_clients['async_client']

def test_chatopenai_initialization_custom_params(mock_openai_clients):
    """Test ChatOpenAI initializes with custom model and temperature."""
    chat_model = ChatOpenAI(model='gpt-4', temperature=0.1, openai_api_key='custom-key')
    assert chat_model.model == 'gpt-4'
    assert chat_model.temperature == 0.1
    mock_openai_clients['sync_constructor'].assert_called_once_with(api_key="custom-key")
    mock_openai_clients['async_constructor'].assert_called_once_with(api_key="custom-key")

def test_chatopenai_initialization_with_max_tokens(mock_openai_clients):
    """Test ChatOpenAI initializes with a specific max_tokens value."""
    chat_model = ChatOpenAI(max_tokens=100, openai_api_key='mt-key')
    assert chat_model.max_tokens == 100
    # Verify client constructors still called correctly
    mock_openai_clients['sync_constructor'].assert_called_once_with(api_key="mt-key")
    mock_openai_clients['async_constructor'].assert_called_once_with(api_key="mt-key")

@pytest.mark.parametrize(
    "input_message, expected_dict",
    [
        (SystemMessage(content="System prompt"), {'role': 'system', 'content': 'System prompt'}),
        (HumanMessage(content="User query"), {'role': 'user', 'content': 'User query'}),
        (AIMessage(content="Assistant response"), {'role': 'assistant', 'content': 'Assistant response'}),
    ]
)
def test_convert_message_to_dict(mock_openai_clients, input_message, expected_dict):
    """Test the conversion of BaseMessage subclasses to OpenAI dict format."""
    # Need an instance to call the method
    chat_model = ChatOpenAI()
    assert chat_model._convert_message_to_dict(input_message) == expected_dict

# --- Invoke / _generate Tests --- #

def test_chatopenai_generate_success(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    """Test successful synchronous generation (_generate)."""
    chat_model = ChatOpenAI()
    result = chat_model._generate(sample_messages)

    assert isinstance(result, AIMessage)
    assert result.content == "Mocked sync response"
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=chat_model.temperature,
        # max_tokens should not be present by default
    )
    # Check that max_tokens was NOT in the call kwargs
    call_args, call_kwargs = mock_openai_clients['sync_create'].call_args
    assert "max_tokens" not in call_kwargs

def test_chatopenai_generate_with_max_tokens(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    """Test synchronous generation passes max_tokens when set."""
    chat_model = ChatOpenAI(max_tokens=50)
    result = chat_model._generate(sample_messages)

    assert isinstance(result, AIMessage)
    assert result.content == "Mocked sync response"
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=chat_model.temperature,
        max_tokens=50 # Check max_tokens is passed
    )

def test_chatopenai_invoke_success(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    """Test successful synchronous invocation (invoke)."""
    chat_model = ChatOpenAI()
    result = chat_model.invoke(sample_messages)

    assert isinstance(result, AIMessage)
    assert result.content == "Mocked sync response"
    mock_openai_clients['sync_create'].assert_called_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=chat_model.temperature,
        # max_tokens should not be present by default
    )
    # Check that max_tokens was NOT in the call kwargs
    call_args, call_kwargs = mock_openai_clients['sync_create'].call_args
    assert "max_tokens" not in call_kwargs

def test_chatopenai_generate_api_error(mock_openai_clients, sample_messages):
    """Test handling of API errors during synchronous generation."""
    mock_openai_clients['sync_create'].side_effect = openai.APIError("API Failed", request=Mock(), body=None)
    chat_model = ChatOpenAI()
    with pytest.raises(openai.APIError, match="API Failed"):
        chat_model._generate(sample_messages)

def test_chatopenai_generate_none_content(mock_openai_clients, sample_messages):
    """Test handling of None content in synchronous response."""
    # Modify the mock response content to be None
    mock_response = mock_openai_clients['sync_create'].return_value
    mock_response.choices[0].message.content = None
    mock_openai_clients['sync_create'].return_value = mock_response

    chat_model = ChatOpenAI()
    with pytest.raises(ValueError, match="OpenAI response content is None"):
        chat_model._generate(sample_messages)


# --- AInvoke / _agenerate Tests --- #

@pytest.mark.asyncio
async def test_chatopenai_agenerate_success(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    """Test successful asynchronous generation (_agenerate)."""
    chat_model = ChatOpenAI()
    result = await chat_model._agenerate(sample_messages)

    assert isinstance(result, AIMessage)
    assert result.content == "Mocked async response"
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=chat_model.temperature,
        # max_tokens should not be present by default
    )
    # Check that max_tokens was NOT in the call kwargs
    call_args, call_kwargs = mock_openai_clients['async_create'].call_args
    assert "max_tokens" not in call_kwargs

@pytest.mark.asyncio
async def test_chatopenai_ainvoke_success(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    """Test successful asynchronous invocation (ainvoke)."""
    chat_model = ChatOpenAI()
    result = await chat_model.ainvoke(sample_messages)

    assert isinstance(result, AIMessage)
    assert result.content == "Mocked async response"
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=chat_model.temperature,
        # max_tokens should not be present by default
    )
    # Check that max_tokens was NOT in the call kwargs
    call_args, call_kwargs = mock_openai_clients['async_create'].call_args
    assert "max_tokens" not in call_kwargs

@pytest.mark.asyncio
async def test_chatopenai_agenerate_with_max_tokens(mock_openai_clients, sample_messages, sample_messages_dict_openai_fmt):
    """Test asynchronous generation passes max_tokens when set."""
    chat_model = ChatOpenAI(max_tokens=75)
    result = await chat_model._agenerate(sample_messages)

    assert isinstance(result, AIMessage)
    assert result.content == "Mocked async response"
    mock_openai_clients['async_create'].assert_awaited_once_with(
        model=chat_model.model,
        messages=sample_messages_dict_openai_fmt,
        temperature=chat_model.temperature,
        max_tokens=75 # Check max_tokens is passed
    )

@pytest.mark.asyncio
async def test_chatopenai_agenerate_api_error(mock_openai_clients, sample_messages):
    """Test handling of API errors during asynchronous generation."""
    mock_openai_clients['async_create'].side_effect = openai.APIError("Async API Failed", request=Mock(), body=None)
    chat_model = ChatOpenAI()
    with pytest.raises(openai.APIError, match="Async API Failed"):
        await chat_model._agenerate(sample_messages)

@pytest.mark.asyncio
async def test_chatopenai_agenerate_none_content(mock_openai_clients, sample_messages):
    """Test handling of None content in asynchronous response."""
    # Modify the mock response content to be None
    # Retrieve the configured mock response object from the AsyncMock
    mock_response = mock_openai_clients['async_create'].return_value
    mock_response.choices[0].message.content = None
    # Reset side_effect if it was set previously, to ensure return_value is used
    mock_openai_clients['async_create'].side_effect = None

    chat_model = ChatOpenAI()
    with pytest.raises(ValueError, match="OpenAI response content is None"):
        await chat_model._agenerate(sample_messages)


# --- Batch Tests (Leverage invoke/ainvoke mocks) --- #

def test_chatopenai_batch_success(mock_openai_clients, sample_messages):
    """Test batch processing delegates to invoke."""
    chat_model = ChatOpenAI()
    inputs = [sample_messages, sample_messages] # Batch of two identical inputs

    # Modify sync mock to return different content per call if needed, or just track calls
    mock_openai_clients['sync_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 2"))]),
    ]

    results = chat_model.batch(inputs)

    assert len(results) == 2
    assert isinstance(results[0], AIMessage)
    assert results[0].content == "Sync Resp 1"
    assert isinstance(results[1], AIMessage)
    assert results[1].content == "Sync Resp 2"
    assert mock_openai_clients['sync_create'].call_count == 2

def test_chatopenai_batch_return_exceptions(mock_openai_clients, sample_messages):
    """Test batch processing with return_exceptions=True."""
    chat_model = ChatOpenAI()
    inputs = [sample_messages, [HumanMessage(content="error trigger")], sample_messages]

    # Setup side effects: success, error, success
    error = ValueError("Batch Error")
    mock_openai_clients['sync_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 1"))]),
        error,
        MagicMock(choices=[MagicMock(message=MagicMock(content="Sync Resp 3"))]),
    ]

    results = chat_model.batch(inputs, return_exceptions=True)

    assert len(results) == 3
    assert isinstance(results[0], AIMessage)
    assert results[0].content == "Sync Resp 1"
    assert results[1] is error
    assert isinstance(results[2], AIMessage)
    assert results[2].content == "Sync Resp 3"
    assert mock_openai_clients['sync_create'].call_count == 3

@pytest.mark.asyncio
async def test_chatopenai_abatch_success(mock_openai_clients, sample_messages):
    """Test async batch processing delegates to ainvoke."""
    chat_model = ChatOpenAI()
    inputs = [sample_messages, sample_messages]

    # Setup async side effects
    mock_openai_clients['async_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 2"))]),
    ]

    results = await chat_model.abatch(inputs)

    assert len(results) == 2
    assert isinstance(results[0], AIMessage)
    assert results[0].content == "Async Resp 1"
    assert isinstance(results[1], AIMessage)
    assert results[1].content == "Async Resp 2"
    assert mock_openai_clients['async_create'].await_count == 2

@pytest.mark.asyncio
async def test_chatopenai_abatch_return_exceptions(mock_openai_clients, sample_messages):
    """Test async batch processing with return_exceptions=True."""
    chat_model = ChatOpenAI()
    inputs = [sample_messages, [HumanMessage(content="error trigger")], sample_messages]

    # Setup async side effects: success, error, success
    error = ValueError("Async Batch Error")
    mock_openai_clients['async_create'].side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 1"))]),
        error,
        MagicMock(choices=[MagicMock(message=MagicMock(content="Async Resp 3"))]),
    ]

    results = await chat_model.abatch(inputs, return_exceptions=True)

    assert len(results) == 3
    assert isinstance(results[0], AIMessage)
    assert results[0].content == "Async Resp 1"
    assert results[1] is error
    assert isinstance(results[2], AIMessage)
    assert results[2].content == "Async Resp 3"
    # asyncio.gather calls all awaitables even if one raises early when return_exceptions=True
    assert mock_openai_clients['async_create'].await_count == 3 