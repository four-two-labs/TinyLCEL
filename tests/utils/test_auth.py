"""Tests for authentication utilities."""

from unittest.mock import patch
from unittest.mock import MagicMock

import pytest

from tinylcel.utils.auth import get_api_key

ENV_VAR_NAME = 'TEST_SERVICE_API_KEY'
SERVICE_NAME = 'TestService'


def test_get_api_key_from_argument() -> None:
    """Test that get_api_key returns the key from the argument if provided."""
    arg_key = 'key_from_arg'
    assert get_api_key(arg_key, ENV_VAR_NAME, SERVICE_NAME) == arg_key


@patch('os.getenv')
def test_get_api_key_from_env_variable(mock_getenv: MagicMock) -> None:
    """Test that get_api_key returns the key from env var if arg is None."""
    env_key = 'key_from_env'
    mock_getenv.return_value = env_key

    assert get_api_key(None, ENV_VAR_NAME, SERVICE_NAME) == env_key
    mock_getenv.assert_called_once_with(ENV_VAR_NAME)


@patch('os.getenv')
def test_get_api_key_arg_takes_precedence(mock_getenv: MagicMock) -> None:
    """Test that the argument key takes precedence over environment variable."""
    arg_key = 'key_from_arg'
    env_key = 'key_from_env'
    mock_getenv.return_value = env_key

    assert get_api_key(arg_key, ENV_VAR_NAME, SERVICE_NAME) == arg_key
    mock_getenv.assert_not_called()  # Should not be called if arg_key is present


@patch('os.getenv')
def test_get_api_key_raises_value_error_if_not_found(mock_getenv: MagicMock) -> None:
    """Test that get_api_key raises ValueError if no key is found."""
    mock_getenv.return_value = None  # No environment variable

    with pytest.raises(ValueError, match='API key not found') as excinfo:
        get_api_key(None, ENV_VAR_NAME, SERVICE_NAME)

    assert SERVICE_NAME in str(excinfo.value)
    assert ENV_VAR_NAME in str(excinfo.value)
    assert 'API key not found' in str(excinfo.value)
    mock_getenv.assert_called_once_with(ENV_VAR_NAME)


@patch('os.getenv')
def test_get_api_key_empty_arg_uses_env(mock_getenv: MagicMock) -> None:
    """Test that an empty string argument is treated as no argument, falling back to env var."""
    # This behavior depends on how get_api_key treats an empty string.
    # Current implementation: if api_key_arg: so empty string is False-y.
    env_key = 'key_from_env'
    mock_getenv.return_value = env_key

    assert get_api_key('', ENV_VAR_NAME, SERVICE_NAME) == env_key
    mock_getenv.assert_called_once_with(ENV_VAR_NAME)


@patch('os.getenv')
def test_get_api_key_empty_arg_and_no_env_raises_error(mock_getenv: MagicMock) -> None:
    """Test ValueError if empty string arg and no env var."""
    mock_getenv.return_value = None
    with pytest.raises(ValueError, match='API key not found'):
        get_api_key('', ENV_VAR_NAME, SERVICE_NAME)
    mock_getenv.assert_called_once_with(ENV_VAR_NAME)
