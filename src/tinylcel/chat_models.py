"""Base classes for chat models and messages in TinyLCEL."""

import abc
from dataclasses import field
from dataclasses import dataclass

from pydantic import BaseModel

from tinylcel.messages import AIMessage
from tinylcel.runnable import RunnableBase
from tinylcel.messages import MessagesInput


@dataclass(frozen=True)
class BaseMessage:
    """Base class for messages in a chat conversation."""

    content: str
    role: str = field(init=False)  # Role is determined by subclass


@dataclass(frozen=True)
class HumanMessage(BaseMessage):
    """A message from the human user."""

    role: str = field(default='human', init=False)


@dataclass(frozen=True)
class SystemMessage(BaseMessage):
    """A message setting the context for the AI assistant (system prompt)."""

    role: str = field(default='system', init=False)


class BaseChatModel(RunnableBase[MessagesInput, AIMessage], abc.ABC):
    """
    Abstract base class for language models that expose a chat interface.

    This class provides the core functionality for chat models within the
    Tiny LangChain Expression Language (TinyLCEL) framework. It inherits
    from `RunnableBase`, defining the input type as a list of `BaseMessage`
    objects (`MessagesInput`) and the output type as a single `AIMessage`.

    Subclasses must implement the `_generate` and `_agenerate` methods
    to provide the specific logic for interacting with the underlying
    language model API synchronously and asynchronously, respectively.

    The `invoke` and `ainvoke` methods are implemented here, calling the
    respective `_generate` and `_agenerate` methods.
    """

    @abc.abstractmethod
    def _generate(self, messages: MessagesInput) -> AIMessage:
        """
        Synchronous generation method.

        This method should be implemented by subclasses to handle the synchronous
        call to the underlying chat model API.

        Args:
            messages: A list of `BaseMessage` objects representing the
                conversation history and prompt.

        Returns:
            An `AIMessage` containing the model's response.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            Exception: Any exception raised by the underlying API call.
        """
        ...

    @abc.abstractmethod
    async def _agenerate(self, messages: MessagesInput) -> AIMessage:
        """
        Asynchronous generation method.

        This method should be implemented by subclasses to handle the asynchronous
        call to the underlying chat model API.

        Args:
            messages: A list of `BaseMessage` objects representing the
                conversation history and prompt.

        Returns:
            An awaitable resolving to an `AIMessage` containing the model's response.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            Exception: Any exception raised by the underlying API call.
        """
        ...

    def invoke(self, input: MessagesInput) -> AIMessage:
        """
        Invoke the chat model synchronously.

        This method takes a list of messages, calls the `_generate` method
        implemented by the subclass, and returns the resulting AI message.

        Args:
            input: A list of `BaseMessage` objects.

        Returns:
            An `AIMessage` representing the model's response.
        """
        return self._generate(input)

    async def ainvoke(self, input: MessagesInput) -> AIMessage:
        """
        Invoke the chat model asynchronously.

        This method takes a list of messages, calls the `_agenerate` method
        implemented by the subclass, and returns the resulting AI message.

        Args:
            input: A list of `BaseMessage` objects.

        Returns:
            An awaitable resolving to an `AIMessage` representing the model's
            response.
        """
        return await self._agenerate(input)

    @abc.abstractmethod
    def with_structured_output[T: BaseModel](self, schema: type[T]) -> 'StructuredBaseChatModel[T]':
        """
        Return a new runnable that outputs structured data using the given Pydantic schema.

        This method creates a new runnable that combines this chat model with a
        PydanticOutputParser to ensure the output conforms to the specified schema.
        Subclasses may override this method to use provider-specific structured
        output capabilities (e.g., OpenAI's response_format parameter).

        Args:
            schema: A Pydantic BaseModel class that defines the expected output structure.

        Returns:
            A new runnable that outputs instances of the specified Pydantic model.

        Example:
            ```python
            from pydantic import BaseModel
            from tinylcel.chat_models.openai import ChatOpenAI
            from tinylcel.messages import HumanMessage


            class Person(BaseModel):
                name: str
                age: int


            model = ChatOpenAI()
            structured_model = model.with_structured_output(Person)
            result = structured_model.invoke([HumanMessage(content='Tell me about John who is 25 years old')])
            # result is now a Person instance
            print(result.name)  # "John"
            print(result.age)  # 25
            ```
        """
        ...


@dataclass
class StructuredBaseChatModel[T: BaseModel](BaseChatModel, RunnableBase[MessagesInput, T], abc.ABC):
    """
    Abstract base class for chat models that expose a chat interface and support structured output.

    This class provides the core functionality for chat models within the
    Tiny LangChain Expression Language (TinyLCEL) framework. It inherits
    from `BaseChatModel`, defining the output type as a Pydantic model.
    """

    output_type: type[T] = field(default=None)  # type: ignore[assignment]

    @abc.abstractmethod
    def _generate(self, messages: MessagesInput) -> T:  # type: ignore[override]
        """
        Generate a response from the chat model.

        This method takes a list of messages, calls the `_generate` method
        implemented by the subclass, and returns the resulting AI message.
        """
        ...

    @abc.abstractmethod
    async def _agenerate(self, messages: MessagesInput) -> T:  # type: ignore[override]
        """Generate a response from the chat model asynchronously."""
        ...

    def invoke(self, input: MessagesInput) -> T:  # type: ignore[override]
        """Invoke the chat model synchronously."""
        return self._generate(input)

    async def ainvoke(self, input: MessagesInput) -> T:  # type: ignore[override]
        """Invoke the chat model asynchronously."""
        return await self._agenerate(input)
