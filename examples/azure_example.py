import os
import asyncio

from dotenv import load_dotenv

from tinylcel.prompts import ChatPromptTemplate
from tinylcel.output_parsers import StrOutputParser
from tinylcel.chat_models.openai import AzureChatOpenAI


async def main():
    load_dotenv()    
    llm = AzureChatOpenAI(
        model=os.getenv('AZURE_OPENAI_MODEL'),
        openai_api_key=os.getenv('AZURE_OPENAI_APIKEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        temperature=0.75,
    )

    chain = (
        ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful assistant that can answer questions.'),
            ('human', 'Tell me a single interesting fact about {topic}.'),
        ])
        | llm 
        | StrOutputParser()
    )

    input_data = {
        'topic': 'black holes'
    }

    print('--- Synchronous Invocation ---')
    sync_result = chain.invoke(input_data)
    print(f'Input Topic: {input_data['topic']}')
    print(f'Final Output Type: {type(sync_result)}')
    print(f'Final Output:\n{sync_result}')
    print('-----------------------------\n')

    print('--- Asynchronous Invocation ---')
    async_result = await chain.ainvoke(input_data)
    print(f'Input Topic: {input_data['topic']}')
    print(f'Final Output Type: {type(async_result)}')
    print(f'Final Output:\n{async_result}')
    print('------------------------------')


if __name__ == '__main__':
    asyncio.run(main())