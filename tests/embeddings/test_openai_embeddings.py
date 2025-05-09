# This is a placeholder for the content of tests/embeddings/test_openai.py which will be moved here.
# The actual file content will be read and then written to this new filename. 

"""Tests for OpenAI embedding models."""

import pytest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import openai
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.embedding import Embedding
from openai._types import NOT_GIVEN

from tinylcel.embeddings.openai import OpenAIEmbeddings

# Default model used in tests if not specified otherwise
TEST_MODEL = 'text-embedding-ada-002'

@pytest.fixture
def mock_openai_client():
    """Fixture for mocking openai.OpenAI."""
    with patch('tinylcel.embeddings.openai.openai.OpenAI', autospec=True) as mock_client_constructor:
        mock_client_instance = mock_client_constructor.return_value
        mock_client_instance.embeddings = MagicMock()
        mock_client_instance.embeddings.create = MagicMock()
        yield mock_client_constructor

@pytest.fixture
def mock_openai_async_client():
    """Fixture for mocking openai.AsyncOpenAI."""
    with patch('tinylcel.embeddings.openai.openai.AsyncOpenAI', autospec=True) as mock_async_client_constructor:
        mock_async_client_instance = mock_async_client_constructor.return_value
        mock_async_client_instance.embeddings = MagicMock()
        mock_async_client_instance.embeddings.create = AsyncMock()
        yield mock_async_client_constructor

@pytest.fixture
def mock_get_openai_api_key():
    """Fixture to mock _get_openai_api_key. 
    Use @patch('tinylcel.embeddings.openai._get_openai_api_key') in specific tests if needed directly.
    The OpenAIEmbeddings class uses this internally.
    """
    pass

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_openai_embeddings_init_with_api_key_arg(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test OpenAIEmbeddings initialization with API key as argument."""
    api_key_val = 'test_api_key_arg'
    mock_get_key_helper.return_value = api_key_val

    embeddings = OpenAIEmbeddings(openai_api_key=api_key_val, model=TEST_MODEL)
    
    mock_get_key_helper.assert_called_once_with(api_key_val)
    assert embeddings.openai_api_key == api_key_val
    mock_openai_client.assert_called_once_with(
        api_key=api_key_val, max_retries=openai.DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_async_client.assert_called_once_with(
        api_key=api_key_val, max_retries=openai.DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_openai_embeddings_init_with_env_var(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test OpenAIEmbeddings initialization with API key from environment variable."""
    env_api_key = 'test_api_key_env'
    mock_get_key_helper.return_value = env_api_key
    
    embeddings = OpenAIEmbeddings(model=TEST_MODEL, openai_api_key=None)
    
    mock_get_key_helper.assert_called_once_with(None)
    assert embeddings.openai_api_key is None
    mock_openai_client.assert_called_once_with(
        api_key=env_api_key, max_retries=openai.DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_async_client.assert_called_once_with(
        api_key=env_api_key, max_retries=openai.DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_openai_embeddings_init_api_key_priority(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test API key argument takes precedence (handled by _get_openai_api_key)."""
    arg_api_key = 'key_from_arg'
    mock_get_key_helper.return_value = arg_api_key

    embeddings = OpenAIEmbeddings(openai_api_key=arg_api_key, model=TEST_MODEL)
    
    mock_get_key_helper.assert_called_once_with(arg_api_key)
    assert embeddings.openai_api_key == arg_api_key
    mock_openai_client.assert_called_once_with(
        api_key=arg_api_key, max_retries=openai.DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )
    mock_openai_async_client.assert_called_once_with(
        api_key=arg_api_key, max_retries=openai.DEFAULT_MAX_RETRIES, timeout=NOT_GIVEN
    )

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_openai_embeddings_init_no_api_key_raises_value_error(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test OpenAIEmbeddings raises ValueError if no API key is found (via helper)."""
    mock_get_key_helper.side_effect = ValueError('OpenAI API key not found.')
    
    with pytest.raises(ValueError) as excinfo:
        OpenAIEmbeddings(model=TEST_MODEL, openai_api_key=None) 
        
    assert 'OpenAI API key not found' in str(excinfo.value)
    mock_get_key_helper.assert_called_once_with(None)
    mock_openai_client.assert_not_called()
    mock_openai_async_client.assert_not_called()

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_openai_embeddings_init_with_custom_params(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test OpenAIEmbeddings initialization with custom base_url, timeout, max_retries."""
    resolved_key = 'test_key' 
    mock_get_key_helper.return_value = resolved_key

    base_url_val = 'https://api.example.com'
    timeout_val = 30.0
    max_retries_val = 5
    
    OpenAIEmbeddings(
        openai_api_key='some_initial_key',
        model=TEST_MODEL, 
        base_url=base_url_val, 
        timeout=timeout_val,
        max_retries=max_retries_val
    )
    
    mock_get_key_helper.assert_called_once_with('some_initial_key')
    mock_openai_client.assert_called_once_with(
        api_key=resolved_key, base_url=base_url_val, max_retries=max_retries_val, timeout=timeout_val
    )
    mock_openai_async_client.assert_called_once_with(
        api_key=resolved_key, base_url=base_url_val, max_retries=max_retries_val, timeout=timeout_val
    )

# Helper to create mock embedding responses
def _get_mock_embedding_response(embeddings: list[list[float]], model: str) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[
            Embedding(embedding=emb, index=i, object='embedding') 
            for i, emb in enumerate(embeddings)
        ],
        model=model,
        object='list',
        usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10)
    )

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_embed_documents_successful(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test embed_documents successfully returns embeddings."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    texts = ['doc1', 'doc2']
    expected_embeddings = [[1.0, 2.0], [3.0, 4.0]]
    model = 'text-embedding-3-small'

    mock_embeddings_create = mock_openai_client.return_value.embeddings.create
    mock_embeddings_create.return_value = _get_mock_embedding_response(expected_embeddings, model)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=model)
    result = embeddings_model.embed_documents(texts)

    assert result == expected_embeddings
    mock_embeddings_create.assert_called_once_with(model=model, input=texts)

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_embed_documents_with_dimensions(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test embed_documents with custom dimensions."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    texts = ['doc1']
    expected_embeddings = [[0.1, 0.2]]
    model = 'text-embedding-3-large'
    dimensions = 2

    mock_embeddings_create = mock_openai_client.return_value.embeddings.create
    mock_embeddings_create.return_value = _get_mock_embedding_response(expected_embeddings, model)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=model, dimensions=dimensions)
    result = embeddings_model.embed_documents(texts)

    assert result == expected_embeddings
    mock_embeddings_create.assert_called_once_with(model=model, input=texts, dimensions=dimensions)

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_embed_query_successful(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test embed_query successfully returns an embedding."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    text = 'query1'
    expected_embedding = [1.0, 2.0, 3.0]
    model = 'text-embedding-ada-002'

    mock_embeddings_create = mock_openai_client.return_value.embeddings.create
    mock_embeddings_create.return_value = _get_mock_embedding_response([expected_embedding], model)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=model)
    result = embeddings_model.embed_query(text)

    assert result == expected_embedding
    mock_embeddings_create.assert_called_once_with(model=model, input=[text])

@patch('tinylcel.embeddings.openai._get_openai_api_key')
def test_embed_query_api_error(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test embed_query raises APIError on API failure."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    text = 'query1'

    mock_embeddings_create = mock_openai_client.return_value.embeddings.create
    mock_embeddings_create.side_effect = openai.APIError(message="API error", request=MagicMock(), body=None)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=TEST_MODEL)
    with pytest.raises(openai.APIError):
        embeddings_model.embed_query(text)

@pytest.mark.asyncio
@patch('tinylcel.embeddings.openai._get_openai_api_key')
async def test_aembed_documents_successful(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test aembed_documents successfully returns embeddings asynchronously."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    texts = ['doc1_async', 'doc2_async']
    expected_embeddings = [[5.0, 6.0], [7.0, 8.0]]
    model = 'text-embedding-3-small'

    mock_async_embeddings_create = mock_openai_async_client.return_value.embeddings.create
    mock_async_embeddings_create.return_value = _get_mock_embedding_response(expected_embeddings, model)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=model)
    result = await embeddings_model.aembed_documents(texts)

    assert result == expected_embeddings
    mock_async_embeddings_create.assert_awaited_once_with(model=model, input=texts)

@pytest.mark.asyncio
@patch('tinylcel.embeddings.openai._get_openai_api_key')
async def test_aembed_query_successful(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test aembed_query successfully returns an embedding asynchronously."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    text = 'query1_async'
    expected_embedding = [9.0, 10.0]
    model = 'text-embedding-ada-002'

    mock_async_embeddings_create = mock_openai_async_client.return_value.embeddings.create
    mock_async_embeddings_create.return_value = _get_mock_embedding_response([expected_embedding], model)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=model)
    result = await embeddings_model.aembed_query(text)

    assert result == expected_embedding
    mock_async_embeddings_create.assert_awaited_once_with(model=model, input=[text])

@pytest.mark.asyncio
@patch('tinylcel.embeddings.openai._get_openai_api_key')
async def test_aembed_query_api_error(
    mock_get_key_helper: MagicMock,
    mock_openai_client: MagicMock,
    mock_openai_async_client: MagicMock,
) -> None:
    """Test aembed_query raises APIError on API failure asynchronously."""
    resolved_key = 'test_key'
    mock_get_key_helper.return_value = resolved_key
    text = 'query1_async_fail'

    mock_async_embeddings_create = mock_openai_async_client.return_value.embeddings.create
    mock_async_embeddings_create.side_effect = openai.APIError(message="Async API error", request=MagicMock(), body=None)

    embeddings_model = OpenAIEmbeddings(openai_api_key='somekey', model=TEST_MODEL)
    with pytest.raises(openai.APIError):
        await embeddings_model.aembed_query(text) 