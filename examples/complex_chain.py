import asyncio
from typing import Dict

from dotenv import load_dotenv

from tinylcel.runnable import runnable
from tinylcel.messages import HumanMessage
from tinylcel.messages import MessagesInput
from tinylcel.runnable import RunnableLambda
from tinylcel.providers.openai import ChatOpenAI
from tinylcel.output_parsers import StrOutputParser


@runnable
def create_prompt(inputs: Dict[str, str]) -> MessagesInput:
    topic = inputs.get('topic', 'bears')
    prompt = f'Tell me a single interesting fact about {topic}.'
    return [HumanMessage(content=prompt)]


async def main():
    load_dotenv()    
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

    chain = (
        create_prompt 
        | llm 
        | StrOutputParser() 
        | RunnableLambda(lambda s: s.upper())
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