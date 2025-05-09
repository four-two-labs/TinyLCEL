'''Authentication utilities.'''

import os
from typing import Optional

def get_api_key(
    api_key_arg: Optional[str],
    env_var_name: str,
    service_name: str,
) -> str:
    '''Retrieves an API key from an argument or environment variable.

    Args:
        api_key_arg: The API key passed as an argument.
        env_var_name: The name of the environment variable to check.
        service_name: The name of the service for error messages.

    Returns:
        The API key.

    Raises:
        ValueError: If the API key is not found.
    '''
    if api_key_arg:
        return api_key_arg
    
    env_api_key = os.getenv(env_var_name)
    if env_api_key:
        return env_api_key
    
    raise ValueError(
        f'{service_name} API key not found. Please set the {env_var_name} ' +
        'environment variable or pass the API key argument.'
    ) 