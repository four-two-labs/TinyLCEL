"""Example demonstrating structured output with ChatOpenAI using OpenAI's parse method.

This example showcases the with_structured_output() method which uses OpenAI's
beta.chat.completions.parse() API to return parsed Pydantic models directly,
eliminating the need for additional JSON parsing steps and providing more
reliable structured outputs.
"""

from pydantic import BaseModel

from tinylcel.messages import HumanMessage
from tinylcel.messages import SystemMessage
from tinylcel.providers.openai.chat_models import ChatOpenAI


class Person(BaseModel):
    """A person with basic information."""
    name: str
    age: int
    city: str
    occupation: str


class Step(BaseModel):
    """A math problem solving step."""
    explanation: str
    output: str


class MathResponse(BaseModel):
    """A complete math problem solution with steps."""
    steps: list[Step]
    final_answer: str


class BookRecommendation(BaseModel):
    """A book recommendation with details."""
    title: str
    author: str
    genre: str
    rating: float
    summary: str


def main() -> None:
    """Demonstrate structured output functionality."""
    # Initialize the chat model - using gpt-4o-mini which supports structured output
    model = ChatOpenAI(model='gpt-4o-mini')

    # Example 1: Person extraction
    print('=== Example 1: Person Information Extraction ===')
    person_model = model.with_structured_output(Person)

    person_result = person_model.invoke([
        SystemMessage(content='You are a helpful assistant that extracts information from a text.'),
        HumanMessage(content=(
            'Tell me about Sarah, a 28-year-old software engineer living in San Francisco.'
        ))
    ])

    print(f'Name: {person_result.name}')
    print(f'Age: {person_result.age}')
    print(f'City: {person_result.city}')
    print(f'Occupation: {person_result.occupation}')
    print(f'Type: {type(person_result)}')
    print()

    # Example 2: Math problem solving with steps (complex nested structure)
    print('=== Example 2: Math Problem Solving ===')
    math_model = model.with_structured_output(MathResponse)

    math_result = math_model.invoke([
        SystemMessage(content='You are a math tutor. You will be given a math problem and you will need to solve it step by step.'),
        HumanMessage(content='Solve the equation: 3x + 7 = 16')
    ])

    print('Steps:')
    for i, step in enumerate(math_result.steps, 1):
        print(f'  {i}. {step.explanation}')
        print(f'     Output: {step.output}')
    print(f'Final Answer: {math_result.final_answer}')
    print(f'Type: {type(math_result)}')
    print()

    # Example 3: Book recommendation
    print('=== Example 3: Book Recommendation ===')
    book_model = model.with_structured_output(BookRecommendation)

    book_result = book_model.invoke([
        HumanMessage(content=(
            'Recommend a good science fiction book for someone who enjoys '
            'stories about space exploration and alien contact.'
        ))
    ])

    print(f'Title: {book_result.title}')
    print(f'Author: {book_result.author}')
    print(f'Genre: {book_result.genre}')
    print(f'Rating: {book_result.rating}')
    print(f'Summary: {book_result.summary}')
    print(f'Type: {type(book_result)}')


if __name__ == '__main__':
    main()
