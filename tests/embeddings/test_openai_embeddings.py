from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import openai
import pytest

# Import real client classes for spec argument in from_client tests
from openai import OpenAI
from openai import Timeout
from openai import AsyncOpenAI
from openai import AzureOpenAI
from openai import AsyncAzureOpenAI
from openai._types import NOT_GIVEN
from openai import OpenAI as RealOpenAI
from openai import AsyncOpenAI as RealAsyncOpenAI

from tinylcel.embeddings.openai import OpenAIEmbeddings
from tinylcel.embeddings.openai import AzureOpenAIEmbeddings

# --- Constants for Tests ---
TEST_OPENAI_API_KEY = 'test-openai-key-for-embeddings'
TEST_AZURE_API_KEY = 'test-azure-key-for-embeddings'

OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
OPENAI_EMBEDDING_MODEL_V3_SMALL = 'text-embedding-3-small'

AZURE_ENDPOINT = 'https://test-azure-openai-resource.openai.azure.com'
AZURE_API_VERSION = '2023-07-01-preview'
AZURE_DEPLOYMENT_NAME = 'my-azure-embedding-deployment'

SAMPLE_DOC_TEXTS = ['Hello TinyLCEL world!', 'Testing OpenAI embeddings.']
SAMPLE_QUERY_TEXT = 'What is TinyLCEL?'

# --- Mocked Embedding Data ---
mock_openai_embedding_1 = MagicMock(spec=openai.types.Embedding)
mock_openai_embedding_1.embedding = [0.1, 0.2, 0.3, 0.4]
mock_openai_embedding_2 = MagicMock(spec=openai.types.Embedding)
mock_openai_embedding_2.embedding = [0.5, 0.6, 0.7, 0.8]
mock_openai_create_embedding_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
mock_openai_create_embedding_response.data = [mock_openai_embedding_1, mock_openai_embedding_2]

mock_azure_embedding_1 = MagicMock(spec=openai.types.Embedding)
mock_azure_embedding_1.embedding = [1.1, 1.2, 1.3, 1.4]
mock_azure_embedding_2 = MagicMock(spec=openai.types.Embedding)
mock_azure_embedding_2.embedding = [1.5, 1.6, 1.7, 1.8]
mock_azure_create_embedding_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
mock_azure_create_embedding_response.data = [mock_azure_embedding_1, mock_azure_embedding_2]


# --- Fixtures ---
@pytest.fixture
def mock_openai_clients_for_embeddings() -> Generator[Dict[str, Any], None, None]:
    mock_sync_openai_client = MagicMock(spec=OpenAI)
    mock_sync_openai_client.copy = MagicMock(return_value=MagicMock(spec=OpenAI))
    mock_sync_embeddings_service = MagicMock(spec=openai.resources.Embeddings)
    mock_sync_embeddings_service.create.return_value = mock_openai_create_embedding_response
    mock_sync_openai_client.embeddings = mock_sync_embeddings_service
    mock_async_openai_client = MagicMock(spec=AsyncOpenAI)
    mock_async_openai_client.copy = MagicMock(return_value=MagicMock(spec=AsyncOpenAI))
    mock_async_embeddings_service = MagicMock(spec=openai.resources.AsyncEmbeddings)
    mock_async_create_method = AsyncMock(return_value=mock_openai_create_embedding_response)
    mock_async_embeddings_service.create = mock_async_create_method
    mock_async_openai_client.embeddings = mock_async_embeddings_service
    with patch(
        'tinylcel.embeddings.openai.openai.OpenAI', return_value=mock_sync_openai_client
    ) as mock_sync_constructor:
        with patch(
            'tinylcel.embeddings.openai.openai.AsyncOpenAI', return_value=mock_async_openai_client
        ) as mock_async_constructor:
            yield {
                'sync_client': mock_sync_openai_client,
                'async_client': mock_async_openai_client,
                'sync_create': mock_sync_embeddings_service.create,
                'async_create': mock_async_create_method,
                'sync_constructor': mock_sync_constructor,
                'async_constructor': mock_async_constructor,
            }


@pytest.fixture
def mock_azure_clients_for_embeddings() -> Generator[Dict[str, Any], None, None]:
    mock_sync_azure_client = MagicMock(spec=AzureOpenAI)
    mock_sync_azure_embeddings_service = MagicMock(spec=openai.resources.Embeddings)
    mock_sync_azure_embeddings_service.create.return_value = mock_azure_create_embedding_response
    mock_sync_azure_client.embeddings = mock_sync_azure_embeddings_service
    mock_async_azure_client = MagicMock(spec=AsyncAzureOpenAI)
    mock_async_azure_embeddings_service = MagicMock(spec=openai.resources.AsyncEmbeddings)
    mock_async_azure_create_method = AsyncMock(return_value=mock_azure_create_embedding_response)
    mock_async_azure_embeddings_service.create = mock_async_azure_create_method
    mock_async_azure_client.embeddings = mock_async_azure_embeddings_service
    with patch(
        'tinylcel.embeddings.openai.AzureOpenAI', return_value=mock_sync_azure_client
    ) as mock_sync_azure_constructor:
        with patch(
            'tinylcel.embeddings.openai.AsyncAzureOpenAI', return_value=mock_async_azure_client
        ) as mock_async_azure_constructor:
            yield {
                'sync_client': mock_sync_azure_client,
                'async_client': mock_async_azure_client,
                'sync_create': mock_sync_azure_embeddings_service.create,
                'async_create': mock_async_azure_create_method,
                'sync_constructor': mock_sync_azure_constructor,
                'async_constructor': mock_async_azure_constructor,
            }


@pytest.fixture(autouse=True)
def set_env_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'key-from-env-for-embeddings')
    # Set Azure-specific env vars that the underlying SDK might check
    # if api_key or api_version are not directly passed or are None during AzureOpenAI client init
    monkeypatch.setenv(
        'AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY
    )  # Default for Azure tests if not overridden by delenv
    monkeypatch.setenv('OPENAI_API_VERSION', AZURE_API_VERSION)  # Default for Azure tests if not overridden by delenv


@pytest.fixture
def sample_texts_and_query() -> Tuple[List[str], str]:
    return SAMPLE_DOC_TEXTS, SAMPLE_QUERY_TEXT


# --- Initial Placeholder Test ---
def test_file_setup_placeholder() -> None:
    assert True


# --- OpenAIEmbeddings Initialization Tests ---
def test_openai_embeddings_initialization_defaults(
    mock_openai_clients_for_embeddings: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key-for-openai-init')
    embedder = OpenAIEmbeddings()
    assert embedder.model == OPENAI_EMBEDDING_MODEL
    assert embedder.api_key is None
    assert embedder.base_url is None
    assert embedder.dimensions is None
    assert embedder.max_retries == openai.DEFAULT_MAX_RETRIES
    assert embedder.timeout == NOT_GIVEN
    expected_client_kwargs = {
        'api_key': 'env-key-for-openai-init',
        'max_retries': openai.DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
    }
    mock_openai_clients_for_embeddings['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients_for_embeddings['async_constructor'].assert_called_once_with(**expected_client_kwargs)
    assert embedder._client == mock_openai_clients_for_embeddings['sync_client']
    assert embedder._async_client == mock_openai_clients_for_embeddings['async_client']


def test_openai_embeddings_initialization_explicit_key(mock_openai_clients_for_embeddings: Dict[str, Any]) -> None:
    embedder = OpenAIEmbeddings(api_key=TEST_OPENAI_API_KEY)
    assert embedder.api_key == TEST_OPENAI_API_KEY
    expected_client_kwargs = {
        'api_key': TEST_OPENAI_API_KEY,
        'max_retries': openai.DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
    }
    mock_openai_clients_for_embeddings['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients_for_embeddings['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_openai_embeddings_initialization_custom_params(mock_openai_clients_for_embeddings: Dict[str, Any]) -> None:
    custom_timeout = Timeout(timeout=30.0, connect=5.0)  # openai.Timeout
    embedder = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL_V3_SMALL,
        api_key=TEST_OPENAI_API_KEY,
        base_url='https://custom.openai.com',
        dimensions=512,
        max_retries=5,
        timeout=custom_timeout,
    )
    assert embedder.model == OPENAI_EMBEDDING_MODEL_V3_SMALL
    assert embedder.base_url == 'https://custom.openai.com'
    assert embedder.dimensions == 512
    assert embedder.max_retries == 5
    assert embedder.timeout == custom_timeout
    expected_client_kwargs = {
        'api_key': TEST_OPENAI_API_KEY,
        'base_url': 'https://custom.openai.com',
        'max_retries': 5,
        'timeout': custom_timeout,
    }
    mock_openai_clients_for_embeddings['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_openai_clients_for_embeddings['async_constructor'].assert_called_once_with(**expected_client_kwargs)


# --- OpenAIEmbeddings Method Tests ---
def test_openai_embeddings_prepare_kwargs(sample_texts_and_query: Tuple[List[str], str]) -> None:
    texts, _ = sample_texts_and_query
    embedder_no_dims = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_V3_SMALL, api_key='dummy')
    kwargs_no_dims = embedder_no_dims._prepare_create_embedding_kwargs(texts)
    assert kwargs_no_dims == {'model': OPENAI_EMBEDDING_MODEL_V3_SMALL, 'input': texts}
    embedder_with_dims = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_V3_SMALL, dimensions=512, api_key='dummy')
    kwargs_with_dims = embedder_with_dims._prepare_create_embedding_kwargs(texts)
    assert kwargs_with_dims == {'model': OPENAI_EMBEDDING_MODEL_V3_SMALL, 'input': texts, 'dimensions': 512}


def test_openai_embed_documents_success(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    texts, _ = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=TEST_OPENAI_API_KEY)
    embeddings = embedder.embed_documents(texts)
    assert embeddings == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    mock_openai_clients_for_embeddings['sync_create'].assert_called_once_with(
        model=OPENAI_EMBEDDING_MODEL, input=texts
    )


def test_openai_embed_query_success(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    _, query = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=TEST_OPENAI_API_KEY)
    mock_single_embedding = MagicMock(spec=openai.types.Embedding)
    mock_single_embedding.embedding = [0.1, 0.2, 0.3, 0.4]
    mock_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
    mock_response.data = [mock_single_embedding]
    mock_openai_clients_for_embeddings['sync_create'].return_value = mock_response
    embedding = embedder.embed_query(query)
    assert embedding == [0.1, 0.2, 0.3, 0.4]
    mock_openai_clients_for_embeddings['sync_create'].assert_called_once_with(
        model=OPENAI_EMBEDDING_MODEL, input=[query]
    )


def test_openai_embed_documents_with_dimensions(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    texts, _ = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_V3_SMALL, dimensions=256, api_key=TEST_OPENAI_API_KEY)
    embedder.embed_documents(texts)
    mock_openai_clients_for_embeddings['sync_create'].assert_called_once_with(
        model=OPENAI_EMBEDDING_MODEL_V3_SMALL, input=texts, dimensions=256
    )


def test_openai_embed_query_with_dimensions(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    _, query = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_V3_SMALL, dimensions=1024, api_key=TEST_OPENAI_API_KEY)
    mock_single_embedding = MagicMock(spec=openai.types.Embedding)
    mock_single_embedding.embedding = [0.9, 0.8, 0.7, 0.6]
    mock_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
    mock_response.data = [mock_single_embedding]
    mock_openai_clients_for_embeddings['sync_create'].return_value = mock_response
    embedder.embed_query(query)
    mock_openai_clients_for_embeddings['sync_create'].assert_called_once_with(
        model=OPENAI_EMBEDDING_MODEL_V3_SMALL, input=[query], dimensions=1024
    )


def test_openai_embed_documents_api_error(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    texts, _ = sample_texts_and_query
    mock_openai_clients_for_embeddings['sync_create'].side_effect = openai.APIError(
        message='OpenAI API Failed', request=Mock(), body=None
    )
    embedder = OpenAIEmbeddings(api_key=TEST_OPENAI_API_KEY)
    with pytest.raises(openai.APIError, match='OpenAI API Failed'):
        embedder.embed_documents(texts)


def test_openai_embed_query_api_error(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    _, query = sample_texts_and_query
    mock_openai_clients_for_embeddings['sync_create'].side_effect = openai.APIError(
        message='OpenAI Query API Failed', request=Mock(), body=None
    )
    embedder = OpenAIEmbeddings(api_key=TEST_OPENAI_API_KEY)
    with pytest.raises(openai.APIError, match='OpenAI Query API Failed'):
        embedder.embed_query(query)


# --- OpenAIEmbeddings Async Method Tests ---
@pytest.mark.asyncio
async def test_openai_aembed_documents_success(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    texts, _ = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=TEST_OPENAI_API_KEY)
    embeddings = await embedder.aembed_documents(texts)
    assert embeddings == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    mock_openai_clients_for_embeddings['async_create'].assert_awaited_once_with(
        model=OPENAI_EMBEDDING_MODEL, input=texts
    )


@pytest.mark.asyncio
async def test_openai_aembed_query_success(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    _, query = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=TEST_OPENAI_API_KEY)
    mock_single_embedding = MagicMock(spec=openai.types.Embedding)
    mock_single_embedding.embedding = [0.1, 0.2, 0.3, 0.4]
    mock_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
    mock_response.data = [mock_single_embedding]
    mock_openai_clients_for_embeddings['async_create'].return_value = mock_response
    embedding = await embedder.aembed_query(query)
    assert embedding == [0.1, 0.2, 0.3, 0.4]
    mock_openai_clients_for_embeddings['async_create'].assert_awaited_once_with(
        model=OPENAI_EMBEDDING_MODEL, input=[query]
    )


@pytest.mark.asyncio
async def test_openai_aembed_documents_with_dimensions(
    mock_openai_clients_for_embeddings: Dict[str, Any], sample_texts_and_query: Tuple[List[str], str]
) -> None:
    texts, _ = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_V3_SMALL, dimensions=256, api_key=TEST_OPENAI_API_KEY)
    await embedder.aembed_documents(texts)
    mock_openai_clients_for_embeddings['async_create'].assert_awaited_once_with(
        model=OPENAI_EMBEDDING_MODEL_V3_SMALL, input=texts, dimensions=256
    )


@pytest.mark.asyncio
async def test_openai_aembed_query_with_dimensions(mock_openai_clients_for_embeddings, sample_texts_and_query):
    _, query = sample_texts_and_query
    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_V3_SMALL, dimensions=1024, api_key=TEST_OPENAI_API_KEY)
    mock_single_embedding = MagicMock(spec=openai.types.Embedding)
    mock_single_embedding.embedding = [0.9, 0.8, 0.7, 0.6]
    mock_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
    mock_response.data = [mock_single_embedding]
    mock_openai_clients_for_embeddings['async_create'].return_value = mock_response
    await embedder.aembed_query(query)
    mock_openai_clients_for_embeddings['async_create'].assert_awaited_once_with(
        model=OPENAI_EMBEDDING_MODEL_V3_SMALL, input=[query], dimensions=1024
    )


@pytest.mark.asyncio
async def test_openai_aembed_documents_api_error(mock_openai_clients_for_embeddings, sample_texts_and_query):
    texts, _ = sample_texts_and_query
    mock_openai_clients_for_embeddings['async_create'].side_effect = openai.APIError(
        message='Async OpenAI API Failed', request=Mock(), body=None
    )
    embedder = OpenAIEmbeddings(api_key=TEST_OPENAI_API_KEY)
    with pytest.raises(openai.APIError, match='Async OpenAI API Failed'):
        await embedder.aembed_documents(texts)


@pytest.mark.asyncio
async def test_openai_aembed_query_api_error(mock_openai_clients_for_embeddings, sample_texts_and_query):
    _, query = sample_texts_and_query
    mock_openai_clients_for_embeddings['async_create'].side_effect = openai.APIError(
        message='Async OpenAI Query API Failed', request=Mock(), body=None
    )
    embedder = OpenAIEmbeddings(api_key=TEST_OPENAI_API_KEY)
    with pytest.raises(openai.APIError, match='Async OpenAI Query API Failed'):
        await embedder.aembed_query(query)


# --- OpenAIEmbeddings from_client Tests ---
def test_openai_embeddings_from_client(mock_openai_clients_for_embeddings: Dict[str, Any]) -> None:
    from tinylcel.embeddings.openai import from_client

    instance = from_client(
        client=mock_openai_clients_for_embeddings['sync_client'],
        async_client=mock_openai_clients_for_embeddings['async_client'],
        model=OPENAI_EMBEDDING_MODEL,
        dimensions=512,
        max_retries=5,
        timeout=30.0,
    )
    assert isinstance(instance, OpenAIEmbeddings)
    assert instance.model == OPENAI_EMBEDDING_MODEL
    assert instance.dimensions == 512
    assert instance.max_retries == 5
    assert instance.timeout == 30.0
    mock_openai_clients_for_embeddings['sync_client'].copy.assert_called_once_with(timeout=30.0, max_retries=5)
    mock_openai_clients_for_embeddings['async_client'].copy.assert_called_once_with(timeout=30.0, max_retries=5)


# --- AzureOpenAIEmbeddings Tests ---
def test_azure_openai_embeddings_initialization_required_params(
    mock_azure_clients_for_embeddings: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Env vars AZURE_OPENAI_API_KEY and OPENAI_API_VERSION are set by autouse fixture set_env_api_keys
    # AzureOpenAIEmbeddings will use these if api_key/api_version args are not passed to its constructor
    # or if they are None.
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,  # Explicitly providing, but could rely on env via get_api_key if needed
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        # api_key is not passed, so it should be resolved from AZURE_OPENAI_API_KEY by get_api_key
    )
    assert embedder.api_key is None  # Because it was taken from env by get_api_key
    assert embedder.azure_endpoint == AZURE_ENDPOINT
    assert embedder.api_version == AZURE_API_VERSION
    assert embedder.azure_deployment == AZURE_DEPLOYMENT_NAME
    assert embedder.model == AZURE_DEPLOYMENT_NAME
    assert embedder.max_retries == openai.DEFAULT_MAX_RETRIES
    assert embedder.timeout == NOT_GIVEN
    expected_client_kwargs = {
        'api_key': TEST_AZURE_API_KEY,
        'azure_endpoint': AZURE_ENDPOINT,
        'api_version': AZURE_API_VERSION,
        'max_retries': openai.DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
    }
    mock_azure_clients_for_embeddings['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_azure_clients_for_embeddings['async_constructor'].assert_called_once_with(**expected_client_kwargs)
    assert isinstance(embedder._client, MagicMock)
    assert isinstance(embedder._async_client, MagicMock)


def test_azure_openai_embeddings_initialization_explicit_key_and_model(
    mock_azure_clients_for_embeddings: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Explicitly clear AZURE_OPENAI_API_KEY to ensure api_key arg is used by get_api_key
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,  # Explicitly providing api_version
        api_key=TEST_AZURE_API_KEY,
        model='custom-model-for-azure',
    )
    assert embedder.model == 'custom-model-for-azure'
    assert embedder.azure_deployment is None
    expected_kwargs = {
        'api_key': TEST_AZURE_API_KEY,
        'azure_endpoint': AZURE_ENDPOINT,
        'api_version': AZURE_API_VERSION,
        'max_retries': openai.DEFAULT_MAX_RETRIES,
        'timeout': NOT_GIVEN,
    }
    mock_azure_clients_for_embeddings['sync_constructor'].assert_called_once_with(**expected_kwargs)
    mock_azure_clients_for_embeddings['async_constructor'].assert_called_once_with(**expected_kwargs)


def test_azure_openai_embeddings_initialization_deployment_overrides_model(
    mock_azure_clients_for_embeddings: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        model='this-should-be-overridden',
        api_key=TEST_AZURE_API_KEY,
    )
    assert embedder.model == AZURE_DEPLOYMENT_NAME


def test_azure_openai_embeddings_initialization_custom_openai_params(
    mock_azure_clients_for_embeddings: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    custom_timeout = Timeout(timeout=25.0)
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        api_key=TEST_AZURE_API_KEY,
        dimensions=1024,
        max_retries=10,
        timeout=custom_timeout,
    )
    assert embedder.dimensions == 1024
    assert embedder.max_retries == 10
    assert embedder.timeout == custom_timeout
    expected_client_kwargs = {
        'api_key': TEST_AZURE_API_KEY,
        'azure_endpoint': AZURE_ENDPOINT,
        'api_version': AZURE_API_VERSION,
        'max_retries': 10,
        'timeout': custom_timeout,
    }
    mock_azure_clients_for_embeddings['sync_constructor'].assert_called_once_with(**expected_client_kwargs)
    mock_azure_clients_for_embeddings['async_constructor'].assert_called_once_with(**expected_client_kwargs)


def test_azure_openai_embeddings_missing_deployment_and_model_raises_error(
    mock_azure_clients_for_embeddings: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)  # Ensure get_api_key uses direct api_key arg if present
    with pytest.raises(ValueError, match="Either 'azure_deployment' or a valid 'model' name must be provided"):
        AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            api_key=TEST_AZURE_API_KEY,  # api_key provided
            model=None,  # type: ignore[arg-type]
            azure_deployment=None,
        )
    mock_azure_clients_for_embeddings['sync_constructor'].reset_mock()
    with pytest.raises(ValueError, match="Either 'azure_deployment' or a valid 'model' name must be provided"):
        AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            api_key=TEST_AZURE_API_KEY,  # api_key provided
            model='',
            azure_deployment=None,
        )


def test_azure_embed_documents_uses_azure_client_and_deployment_model(
    mock_azure_clients_for_embeddings: Dict[str, Any],
    sample_texts_and_query: Tuple[List[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    texts, _ = sample_texts_and_query
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        api_key=TEST_AZURE_API_KEY,
        dimensions=128,
    )
    embeddings = embedder.embed_documents(texts)
    assert embeddings == [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]]
    mock_azure_clients_for_embeddings['sync_create'].assert_called_once_with(
        model=AZURE_DEPLOYMENT_NAME, input=texts, dimensions=128
    )


@pytest.mark.asyncio
async def test_azure_aembed_query_uses_azure_client_and_deployment_model(
    mock_azure_clients_for_embeddings: Dict[str, Any],
    sample_texts_and_query: Tuple[List[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    _, query = sample_texts_and_query
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        api_key=TEST_AZURE_API_KEY,
    )
    mock_single_azure_embedding = MagicMock(spec=openai.types.Embedding)
    mock_single_azure_embedding.embedding = [1.1, 1.2, 1.3, 1.4]
    mock_response = MagicMock(spec=openai.types.CreateEmbeddingResponse)
    mock_response.data = [mock_single_azure_embedding]
    mock_azure_clients_for_embeddings['async_create'].return_value = mock_response
    embedding = await embedder.aembed_query(query)
    assert embedding == [1.1, 1.2, 1.3, 1.4]
    mock_azure_clients_for_embeddings['async_create'].assert_awaited_once_with(
        model=AZURE_DEPLOYMENT_NAME, input=[query]
    )


def test_azure_embed_documents_api_error(
    mock_azure_clients_for_embeddings: Dict[str, Any],
    sample_texts_and_query: Tuple[List[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    texts, _ = sample_texts_and_query
    mock_azure_clients_for_embeddings['sync_create'].side_effect = openai.APIError(
        message='Azure API Failed', request=Mock(), body=None
    )
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        api_key=TEST_AZURE_API_KEY,
    )
    with pytest.raises(openai.APIError, match='Azure API Failed'):
        embedder.embed_documents(texts)


# --- AzureOpenAIEmbeddings from_client Tests ---
def test_azure_openai_embeddings_from_client(mock_azure_clients_for_embeddings, monkeypatch):
    from tinylcel.embeddings.openai import from_azure_client

    instance = from_azure_client(
        client=mock_azure_clients_for_embeddings['sync_client'],
        async_client=mock_azure_clients_for_embeddings['async_client'],
        model=OPENAI_EMBEDDING_MODEL,
        dimensions=512,
        max_retries=5,
        timeout=30.0,
    )
    assert isinstance(instance, AzureOpenAIEmbeddings)
    assert instance.model == OPENAI_EMBEDDING_MODEL
    assert instance.dimensions == 512
    assert instance.max_retries == 5
    assert instance.timeout == 30.0
    mock_azure_clients_for_embeddings['sync_client'].copy.assert_called_once_with(timeout=30.0, max_retries=5)
    mock_azure_clients_for_embeddings['async_client'].copy.assert_called_once_with(timeout=30.0, max_retries=5)


def test_azure_openai_embeddings_from_client_no_deployment_uses_model(mock_azure_clients_for_embeddings, monkeypatch):
    """Test from_client uses model if azure_deployment is None."""
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', TEST_AZURE_API_KEY)  # Ensure API key for __post_init__
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    mock_sync_azure_client = MagicMock(spec=RealOpenAI)
    mock_async_azure_client = MagicMock(spec=RealAsyncOpenAI)
    mock_sync_azure_client.copy.return_value = mock_sync_azure_client
    mock_async_azure_client.copy.return_value = mock_async_azure_client

    instance = AzureOpenAIEmbeddings.from_client(
        client=mock_sync_azure_client, async_client=mock_async_azure_client, model=OPENAI_EMBEDDING_MODEL_V3_SMALL
    )
    assert instance.model == OPENAI_EMBEDDING_MODEL_V3_SMALL
    assert instance.azure_deployment is None
    # Check that the constructors from the fixture were called by __post_init__.
    mock_azure_clients_for_embeddings['sync_constructor'].assert_called_once()
    mock_azure_clients_for_embeddings['async_constructor'].assert_called_once()
