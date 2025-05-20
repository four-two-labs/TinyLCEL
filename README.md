# TinyLCEL

A tiny reproduction of the LangChain Expression Language (LCEL), focusing on the core expression chaining functionality without all the bells and whistles. TinyLCEL provides a simple, composable framework for defining, chaining, and executing chat models, prompt templates, embeddings, and output parsers in Python.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
  - [Messages](#messages)
  - [Runnables and Chaining](#runnables-and-chaining)
  - [Prompt Templates](#prompt-templates)
  - [Chat Models](#chat-models)
  - [Embeddings](#embeddings)
  - [Output Parsers](#output-parsers)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)

## Features

- Simple, composable runnable framework for chaining operations
- Chat model integrations for OpenAI and Azure OpenAI Service
- Embedding models for OpenAI, Azure OpenAI Service, and Cohere
- Prompt templating for chat interactions
- Output parsers for string, JSON, and YAML
- Utility functions for batching, async iteration, and more

## Installation

Requires Python 3.12 or higher.

Install the core package:

```bash
pip install tinylcel
```

Install optional dependencies:

```bash
pip install tinylcel[openai]    # OpenAI chat and embeddings
pip install tinylcel[cohere]    # Cohere embeddings
pip install tinylcel[openai,cohere]  # Both
pip install tinylcel[dev]       # Development dependencies (testing, linting)
```

## Quick Start

```python
import asyncio
from dotenv import load_dotenv
from tinylcel.messages import HumanMessage
from tinylcel.chat_models.openai import ChatOpenAI
from tinylcel.output_parsers import StrOutputParser

async def main():
    load_dotenv()  # Load environment variables from .env
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)

    chain = llm | StrOutputParser()

    messages = [HumanMessage(content='Hello, world!')]
    result = chain.invoke(messages)
    print(result)

if __name__ == '__main__':
    asyncio.run(main())
```

## Key Concepts

### Messages

Defines basic message types for chat interactions:

- `HumanMessage(content: str)` — message from the user
- `AIMessage(content: str)` — message from the AI assistant
- `SystemMessage(content: str)` — system-level context message

Each message has a `.role` attribute (`'human'`, `'ai'`, `'system'`) and `.content`.

### Runnables and Chaining

Components implement the `Runnable` protocol and can be chained using the `|` operator:

```python
from tinylcel.runnable import runnable

@runnable
def reverse_text(text: str) -> str:
    return text[::-1]

chain = reverse_text | str.upper
output = chain.invoke('hello')
# output == 'OLLEH'
```

### Prompt Templates

`ChatPromptTemplate` helps format chat prompts with variables:

```python
from tinylcel.prompts import ChatPromptTemplate
from tinylcel.chat_models.openai import ChatOpenAI
from tinylcel.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', 'What is the capital of {country}?')
])

chain = template | ChatOpenAI(model='gpt-4') | StrOutputParser()
result = chain.invoke({'country': 'France'})
print(result)  # 'Paris'
```

### Chat Models

- `ChatOpenAI` — OpenAI Chat Completions (e.g., GPT-3.5, GPT-4)
- `AzureChatOpenAI` — Azure OpenAI Service chat models

Authentication via environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY`, with additional vars for Azure:
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`.

### Embeddings

- `OpenAIEmbeddings` — embeddings via OpenAI API (`OPENAI_API_KEY`)
- `AzureOpenAIEmbeddings` — Azure OpenAI embeddings
- `CohereEmbeddings` — embeddings via Cohere API (`COHERE_API_KEY`)

### Output Parsers

Provide structured parsing of LLM outputs:

- `StrOutputParser` — extract raw string
- `JsonOutputParser` — parse JSON (handles code fences)
- `YamlOutputParser` — parse YAML (handles code fences)

## Examples

Explore the [`examples/`](./examples/) directory for ready-to-run scripts:

- `simple_chain.py`
- `prompt_template.py`
- `complex_chain.py`
- `embeddings_openai.py`
- `embeddings_cohere.py`
- `chat_openai_from_client.py`, etc.

## Testing

Install development dependencies and run tests:

```bash
pip install tinylcel[dev]
pytest
```

## Contributing

Contributions welcome! Please open issues or pull requests.

### Development

- Run code quality checks: `ruff .`
- Run type checks: `mypy src/`

## License

This project is licensed under the MIT License.

```

</rewritten_file>