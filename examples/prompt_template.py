import asyncio

from dotenv import load_dotenv

from tinylcel.prompts import ChatPromptTemplate
from tinylcel.providers.openai import ChatOpenAI
from tinylcel.output_parsers import StrOutputParser


async def main() -> None:
    load_dotenv()
    llm = ChatOpenAI(model='gpt-4o', temperature=0.75)

    chain = (
        ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful assistant that can answer questions.'),
            ('human', 'Tell me a single interesting fact about {topic}.'),
        ])
        | llm
        | StrOutputParser()
    )

    input_data = {
        'topic': 'the history of Python programming language'
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
