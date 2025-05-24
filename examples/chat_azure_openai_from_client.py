"""Example of using standalone from_azure_client() function."""

import os
import asyncio
from typing import List

import openai
from dotenv import load_dotenv
import httpx  # httpx.URL is used by openai models for base_url

from tinylcel.messages import BaseMessage
from tinylcel.messages import HumanMessage
from tinylcel.chat_models.openai import from_azure_client


async def main() -> None:
    load_dotenv()
    print("--- Example: from_azure_client() ---")

    azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") # Often shared with Azure
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")


    print(f"azure_deployment_name: {azure_deployment_name}")
    print(f"azure_endpoint: {azure_endpoint}")
    print(f"azure_api_version: {azure_api_version}")
    print(f"azure_api_key: {azure_api_key}")

    if not all([azure_deployment_name, azure_endpoint, azure_api_version, azure_api_key]):
        print("One or more Azure environment variables are not set.")
        print("Please set: AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION, AZURE_OPENAI_API_KEY")
        return
    # Narrow types for mypy: ensure values are not None
    assert azure_deployment_name is not None
    assert azure_endpoint is not None
    assert azure_api_version is not None
    assert azure_api_key is not None

    try:
        # 1. Create pre-configured Azure OpenAI clients
        # The SDK will use env vars if parameters like api_key, azure_endpoint, api_version are not passed directly.
        original_sync_client = openai.AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,  # type: ignore[arg-type] # We ensured it's not None
            azure_deployment=azure_deployment_name,
        )
        original_async_client = openai.AsyncAzureOpenAI(
            api_key=azure_api_key,
            azure_deployment=azure_deployment_name,
            api_version=azure_api_version, # type: ignore[arg-type]
            azure_endpoint=azure_endpoint   # type: ignore[arg-type]
        )
        print("Successfully initialized original Azure OpenAI clients.")
        # For display, attempt to get api_version; from_client will use getattr
        sync_client_api_version_display = getattr(original_sync_client, 'api_version', os.getenv("OPENAI_API_VERSION"))
        print(f"Client using endpoint: {original_sync_client.base_url}, version: {sync_client_api_version_display}\n")

        # 2. Instantiate AzureChatOpenAI using from_client
        # model can be the deployment name. azure_deployment param is also available.
        del os.environ['AZURE_OPENAI_API_KEY']
        del os.environ['AZURE_OPENAI_ENDPOINT']
        del os.environ['AZURE_OPENAI_API_VERSION']
        del os.environ['AZURE_OPENAI_DEPLOYMENT']

        chat_model = from_azure_client(
            client=original_sync_client,
            async_client=original_async_client,
            model=str(azure_deployment_name),
            temperature=0.6,
            max_tokens=128
        )
        print(f"AzureChatOpenAI instance created for deployment: {chat_model.model}")

        # 3. Use the chat model
        messages: List[BaseMessage] = [HumanMessage(content="What are the benefits of using Azure OpenAI?")]
        
        print("\nInvoking Azure model synchronously...")
        response_sync = chat_model.invoke(messages)
        print(f"Sync Response: {response_sync.content}")

        print("\nInvoking Azure model asynchronously...")
        response_async = await chat_model.ainvoke(messages)
        print(f"Async Response: {response_async.content}")

    except openai.APIError as e:
        print(f"Azure OpenAI API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'original_async_client' in locals() and original_async_client:
            await original_async_client.close()
        if 'original_sync_client' in locals() and original_sync_client:
            original_sync_client.close()
        print("\nOriginal Azure clients closed if they were initialized.")


if __name__ == "__main__":
    asyncio.run(main()) 