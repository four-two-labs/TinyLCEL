"""Example of using standalone from_client() function."""

import asyncio
from typing import List

import openai
from dotenv import load_dotenv

from tinylcel.messages import BaseMessage
from tinylcel.messages import HumanMessage
from tinylcel.chat_models.openai import from_client


async def main() -> None:
    load_dotenv()
    print("--- Example: from_client() ---")

    try:
        # 1. Create pre-configured OpenAI clients
        original_sync_client = openai.OpenAI()
        original_async_client = openai.AsyncOpenAI()
        print("Successfully initialized original OpenAI clients.")

        # 2. Instantiate ChatOpenAI using standalone from_client function
        chat_model = from_client(
            client=original_sync_client,
            async_client=original_async_client,
            model='gpt-4o-mini',
            temperature=0.5
        )
        print(f"ChatOpenAI instance created with model: {chat_model.model}")

        # 3. Use the chat model
        messages: List[BaseMessage] = [HumanMessage(content="Why is the sky blue?")]
        
        print("\nInvoking synchronously...")
        response_sync = chat_model.invoke(messages)
        print(f"Sync Response: {response_sync.content}")

        print("\nInvoking asynchronously...")
        response_async = await chat_model.ainvoke(messages)
        print(f"Async Response: {response_async.content}")

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        print("Ensure your OPENAI_API_KEY is set and valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure clients are closed if they were successfully created
        if 'original_async_client' in locals() and original_async_client:
            await original_async_client.close()
        if 'original_sync_client' in locals() and original_sync_client:
            original_sync_client.close()
        print("\nOriginal clients closed if they were initialized.")


if __name__ == "__main__":
    asyncio.run(main()) 