"""Tests for Cohere embedding models."""

from typing import List
from typing import Literal
from typing import Generator
from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import cohere  # type: ignore[import-not-found]
import cohere.errors  # type: ignore[import-not-found]
from cohere.client import OMIT  # type: ignore[import-not-found]

from tinylcel.embeddings.cohere import CohereEmbeddings


@pytest.fixture
def mock_cohere_client_v2() -> Generator[MagicMock, None, None]:
    """Fixture for mocking cohere.ClientV2."""
    with patch('cohere.ClientV2', autospec=True) as mock_client_constructor:
        mock_client_instance = mock_client_constructor.return_value
        mock_client_instance.embed = MagicMock()
        yield mock_client_constructor


@pytest.fixture
def mock_cohere_async_client_v2() -> Generator[MagicMock, None, None]:
    """Fixture for mocking cohere.AsyncClientV2."""
    with patch('cohere.AsyncClientV2', autospec=True) as mock_async_client_constructor:
        mock_async_client_instance = mock_async_client_constructor.return_value
        mock_async_client_instance.embed = AsyncMock()
        yield mock_async_client_constructor


def test_cohere_embeddings_init_with_api_key_arg(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test CohereEmbeddings initialization with API key as argument."""
    api_key = 'test_api_key_arg'
    embeddings = CohereEmbeddings(api_key=api_key)

    assert embeddings.api_key == api_key
    mock_cohere_client_v2.assert_called_once_with(api_key=api_key)
    mock_cohere_async_client_v2.assert_called_once_with(api_key=api_key)


@patch('os.getenv')
def test_cohere_embeddings_init_with_env_var(
    mock_getenv: MagicMock,
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test CohereEmbeddings initialization with API key from environment variable."""
    env_api_key = 'test_api_key_env'
    mock_getenv.return_value = env_api_key

    embeddings = CohereEmbeddings()  # No api_key arg

    mock_getenv.assert_called_once_with('COHERE_API_KEY')
    assert embeddings.api_key is None  # api_key arg was None
    mock_cohere_client_v2.assert_called_once_with(api_key=env_api_key)
    mock_cohere_async_client_v2.assert_called_once_with(api_key=env_api_key)


@patch('os.getenv')
def test_cohere_embeddings_init_api_key_priority(
    mock_getenv: MagicMock,
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test API key argument takes precedence over environment variable."""
    arg_api_key = 'key_from_arg'
    env_api_key = 'key_from_env'
    mock_getenv.return_value = env_api_key  # Env var is set

    embeddings = CohereEmbeddings(api_key=arg_api_key)  # API key arg is provided

    assert embeddings.api_key == arg_api_key
    mock_getenv.assert_not_called()  # Should not check env var if arg is present
    mock_cohere_client_v2.assert_called_once_with(api_key=arg_api_key)
    mock_cohere_async_client_v2.assert_called_once_with(api_key=arg_api_key)


@patch('os.getenv')
def test_cohere_embeddings_init_no_api_key_raises_value_error(
    mock_getenv: MagicMock,
    mock_cohere_client_v2: MagicMock,  # To prevent actual client init
    mock_cohere_async_client_v2: MagicMock,  # To prevent actual client init
) -> None:
    """Test CohereEmbeddings raises ValueError if no API key is found."""
    mock_getenv.return_value = None  # No environment variable

    with pytest.raises(ValueError) as excinfo:
        CohereEmbeddings()  # No api_key arg and no env var

    assert 'Cohere API key not found' in str(excinfo.value)
    assert 'COHERE_API_KEY' in str(excinfo.value)
    mock_cohere_client_v2.assert_not_called()
    mock_cohere_async_client_v2.assert_not_called()


def test_cohere_embeddings_init_with_base_url(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test CohereEmbeddings initialization with a custom base_url."""
    api_key = 'test_api_key'
    base_url = 'https://api.example.com'

    embeddings = CohereEmbeddings(api_key=api_key, base_url=base_url)

    assert embeddings.base_url == base_url
    mock_cohere_client_v2.assert_called_once_with(api_key=api_key, base_url=base_url)
    mock_cohere_async_client_v2.assert_called_once_with(api_key=api_key, base_url=base_url)


# Tests for embed_documents
def test_embed_documents_successful(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,  # Not used but required by fixture chain
) -> None:
    """Test embed_documents successfully returns embeddings."""
    api_key: str = 'test_api_key'
    texts: List[str] = ['doc1', 'doc2']
    expected_embeddings: List[List[float]] = [[1.0, 2.0], [3.0, 4.0]]

    mock_client_instance = mock_cohere_client_v2.return_value
    mock_response = MagicMock()
    mock_response.embeddings = MagicMock()
    mock_response.embeddings.float = expected_embeddings
    mock_client_instance.embed.return_value = mock_response

    embeddings_model = CohereEmbeddings(api_key=api_key)
    result: List[List[float]] = embeddings_model.embed_documents(texts)

    assert result == expected_embeddings
    mock_client_instance.embed.assert_called_once_with(
        texts=texts,
        model=embeddings_model.model,  # default model
        input_type='search_document',
        truncate=embeddings_model.truncate,  # default 'END'
        output_dimension=OMIT,  # default OMIT
        embedding_types=['float'],
    )


def test_embed_documents_with_custom_params(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test embed_documents with custom model, truncate, and dimensions."""
    api_key: str = 'test_api_key'
    texts: List[str] = ['doc1']
    custom_model: str = 'embed-multilingual-v3.0'
    custom_truncate: Literal['NONE', 'START', 'END'] = 'START'
    custom_dimensions: int = 1024
    expected_embeddings: List[List[float]] = [[0.1, 0.2, 0.3]]

    mock_client_instance = mock_cohere_client_v2.return_value
    mock_response = MagicMock()
    mock_response.embeddings = MagicMock()
    mock_response.embeddings.float = expected_embeddings
    mock_client_instance.embed.return_value = mock_response

    embeddings_model = CohereEmbeddings(
        api_key=api_key, model=custom_model, truncate=custom_truncate, dimensions=custom_dimensions
    )
    result: List[List[float]] = embeddings_model.embed_documents(texts)

    assert result == expected_embeddings
    mock_client_instance.embed.assert_called_once_with(
        texts=texts,
        model=custom_model,
        input_type='search_document',
        truncate=custom_truncate,
        output_dimension=custom_dimensions,
        embedding_types=['float'],
    )


def test_embed_documents_cohere_api_error(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test embed_documents raises CohereError on API failure."""
    api_key: str = 'test_api_key'
    texts: List[str] = ['doc1']

    mock_client_instance = mock_cohere_client_v2.return_value
    mock_client_instance.embed.side_effect = cohere.errors.InternalServerError(body={})

    embeddings_model = CohereEmbeddings(api_key=api_key)
    with pytest.raises(cohere.errors.InternalServerError):
        embeddings_model.embed_documents(texts)


# Tests for embed_query
def test_embed_query_successful(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test embed_query successfully returns an embedding."""
    api_key: str = 'test_api_key'
    text: str = 'query1'
    expected_embedding: List[float] = [1.0, 2.0, 3.0]

    mock_client_instance = mock_cohere_client_v2.return_value
    mock_response = MagicMock()
    mock_response.embeddings = MagicMock()
    mock_response.embeddings.float = [expected_embedding]
    mock_client_instance.embed.return_value = mock_response

    embeddings_model = CohereEmbeddings(api_key=api_key)
    result: List[float] = embeddings_model.embed_query(text)

    assert result == expected_embedding
    mock_client_instance.embed.assert_called_once_with(
        texts=[text],
        model=embeddings_model.model,
        input_type='search_query',
        truncate=embeddings_model.truncate,
        output_dimension=OMIT,
        embedding_types=['float'],
    )


def test_embed_query_cohere_api_error(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test embed_query raises CohereError on API failure."""
    api_key: str = 'test_api_key'
    text: str = 'query1'

    mock_client_instance = mock_cohere_client_v2.return_value
    mock_client_instance.embed.side_effect = cohere.errors.InternalServerError(body={})

    embeddings_model = CohereEmbeddings(api_key=api_key)
    with pytest.raises(cohere.errors.InternalServerError):
        embeddings_model.embed_query(text)


# Tests for aembed_documents
@pytest.mark.asyncio
async def test_aembed_documents_successful(
    mock_cohere_client_v2: MagicMock,  # Not used but required
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test aembed_documents successfully returns embeddings asynchronously."""
    api_key: str = 'test_api_key'
    texts: List[str] = ['doc1_async', 'doc2_async']
    expected_embeddings: List[List[float]] = [[5.0, 6.0], [7.0, 8.0]]

    mock_async_client_instance = mock_cohere_async_client_v2.return_value
    mock_response = MagicMock()
    mock_response.embeddings = MagicMock()
    mock_response.embeddings.float = expected_embeddings
    mock_async_client_instance.embed.return_value = mock_response

    embeddings_model = CohereEmbeddings(api_key=api_key)
    result: List[List[float]] = await embeddings_model.aembed_documents(texts)

    assert result == expected_embeddings
    mock_async_client_instance.embed.assert_awaited_once_with(
        texts=texts,
        model=embeddings_model.model,
        input_type='search_document',
        embedding_types=['float'],
        output_dimension=OMIT,
        truncate=embeddings_model.truncate,
    )


@pytest.mark.asyncio
async def test_aembed_documents_cohere_api_error(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test aembed_documents raises CohereError on API failure asynchronously."""
    api_key: str = 'test_api_key'
    texts: List[str] = ['doc1_async_fail']

    mock_async_client_instance = mock_cohere_async_client_v2.return_value
    mock_async_client_instance.embed.side_effect = cohere.errors.InternalServerError(body={})

    embeddings_model = CohereEmbeddings(api_key=api_key)
    with pytest.raises(cohere.errors.InternalServerError):
        await embeddings_model.aembed_documents(texts)


# Tests for aembed_query
@pytest.mark.asyncio
async def test_aembed_query_successful(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test aembed_query successfully returns an embedding asynchronously."""
    api_key: str = 'test_api_key'
    text: str = 'query1_async'
    expected_embedding: List[float] = [9.0, 10.0]

    mock_async_client_instance = mock_cohere_async_client_v2.return_value
    mock_response = MagicMock()
    mock_response.embeddings = MagicMock()
    mock_response.embeddings.float = [expected_embedding]
    mock_async_client_instance.embed.return_value = mock_response

    embeddings_model = CohereEmbeddings(api_key=api_key)
    result: List[float] = await embeddings_model.aembed_query(text)

    assert result == expected_embedding
    mock_async_client_instance.embed.assert_awaited_once_with(
        texts=[text],
        model=embeddings_model.model,
        input_type='search_query',
        embedding_types=['float'],
        output_dimension=OMIT,
        truncate=embeddings_model.truncate,
    )


@pytest.mark.asyncio
async def test_aembed_query_cohere_api_error(
    mock_cohere_client_v2: MagicMock,
    mock_cohere_async_client_v2: MagicMock,
) -> None:
    """Test aembed_query raises CohereError on API failure asynchronously."""
    api_key: str = 'test_api_key'
    text: str = 'query1_async_fail'

    mock_async_client_instance = mock_cohere_async_client_v2.return_value
    mock_async_client_instance.embed.side_effect = cohere.errors.InternalServerError(body={})

    embeddings_model = CohereEmbeddings(api_key=api_key)
    with pytest.raises(cohere.errors.InternalServerError):
        await embeddings_model.aembed_query(text)
