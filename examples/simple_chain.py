import asyncio

from dotenv import load_dotenv

from tinylcel.messages import HumanMessage
from tinylcel.providers.openai import ChatOpenAI
from tinylcel.output_parsers import StrOutputParser


async def main():
    load_dotenv() 
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    chain = (
        llm 
        | StrOutputParser()
    )

    messages = [HumanMessage(content='Tell me a very short, SFW knock-knock joke.')]

    print('--- Synchronous Invocation ---')
    sync_result = chain.invoke(messages)
    print(f'Input: {messages[0].content}')
    print(f'Output Type: {type(sync_result)}')
    print(f'Output:\n{sync_result}')
    print('-----------------------------\n')

    print('--- Asynchronous Invocation ---')
    async_result = await chain.ainvoke(messages)
    print(f'Input: {messages[0].content}')
    print(f'Output Type: {type(async_result)}')
    print(f'Output:\n{async_result}')
    print('------------------------------')

if __name__ == '__main__':
    asyncio.run(main())