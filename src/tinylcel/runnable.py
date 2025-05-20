"""Core primitives for the Tiny LangChain Expression Language (LCEL) imitation.

Defines the Runnable protocol, base classes, and concrete implementations
for sequences and lambda functions, enabling chaining of operations.
"""

import abc
import inspect
import asyncio
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import Callable
from typing import Awaitable
from dataclasses import dataclass
from typing import runtime_checkable

from tinylcel.itools import batch

Input = TypeVar("Input")
Output = TypeVar("Output")
Intermediate = TypeVar("Intermediate")
OtherOutput = TypeVar("OtherOutput")

type WrappableFunc[Input, Output] = (
    Callable[[Input], Output] | Callable[[Input], Awaitable[Output]]
)


@runtime_checkable
class Runnable[Input, Output](Protocol):
    """A unit of computation that can be invoked synchronously or asynchronously.

    This protocol defines the essential interface for all runnable components
    in the TinyLCEL framework. Runnables are designed to be composable.

    Attributes:
        Input: The type of input the runnable accepts.
        Output: The type of output the runnable produces.

    """

    def invoke(self, input: Input) -> Output:
        """Invoke the runnable synchronously.

        Args:
            input: The input to the runnable.
            **kwargs: Additional keyword arguments (ignored by default implementations).

        Returns:
            The result of the runnable's computation.

        """
        ...

    async def ainvoke(self, input: Input) -> Output:
        """Invoke the runnable asynchronously.

        Args:
            input: The input to the runnable.
            **kwargs: Additional keyword arguments (ignored by default implementations).

        Returns:
            An awaitable resolving to the result of the runnable's computation.

        """
        ...

    def batch(
        self,
        inputs: list[Input],
        return_exceptions: bool = False,
        max_concurrency: int | None = None,
    ) -> list[Output | BaseException]:
        """Invoke the runnable on a batch of inputs synchronously.

        Args:
            inputs: A list of inputs to the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            A list containing the results for each input.

        """
        ...

    async def abatch(
        self,
        inputs: list[Input],
        return_exceptions: bool = False,
        max_concurrency: int | None = None,
    ) -> list[Output | BaseException]:
        """Invoke the runnable on a batch of inputs asynchronously.

        Args:
            inputs: A list of inputs to the runnable.
            return_exceptions: If True, return exceptions instead of raising them.
            max_concurrency: The maximum number of concurrent invocations.

        Returns:
            An awaitable resolving to a list containing the results for each input.

        """
        ...

    # stream and astream removed
    # def stream(self, input: Input) -> Iterable[Output]:
    #     ...
    # def astream(self, input: Input) -> AsyncIterable[Output]:
    #     ...


class RunnableBase[Input, Output](Runnable[Input, Output], abc.ABC):
    """Abstract base class providing common functionality for Runnables.

    This class implements the `Runnable` protocol and provides a concrete
    implementation for the `__or__` operator, facilitating the chaining
    of runnables into sequences. Subclasses must implement the core
    `invoke` and `ainvoke` methods.
    """

    @abc.abstractmethod
    def invoke(self, input: Input) -> Output:
        """Abstract method for synchronous invocation.

        Subclasses must implement this method to define their synchronous behavior.

        Args:
            input: The input to the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the computation.

        """
        ...

    @abc.abstractmethod
    async def ainvoke(self, input: Input) -> Output:
        """Abstract method for asynchronous invocation.

        Subclasses must implement this method to define their asynchronous behavior.

        Args:
            input: The input to the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            An awaitable resolving to the result of the computation.

        """
        ...

    # Concrete implementation of __or__
    def __or__[Intermediate, OtherOutput](  # Type parameters for the method
        self: Runnable[Input, Intermediate],
        other: Runnable[Intermediate, OtherOutput],
    ) -> "RunnableBase[Input, OtherOutput]":  # Return the concrete RunnableBase type
        """Compose this runnable with another runnable using the | operator.

        This method allows chaining runnables like `runnable1 | runnable2`,
        creating a `RunnableSequence`.

        Args:
            other: The runnable to compose with. Must implement the Runnable protocol.

        Returns:
            A RunnableSequence representing the composition of self | other, as a RunnableBase.

        Raises:
            TypeError: If 'other' is not a valid Runnable instance.

        """
        if not isinstance(other, Runnable):
            raise TypeError(
                f"Expected second argument to be Runnable, got {type(other)}"
            )

        return RunnableSequence(first=self, second=other)

    # Default batch implementation: iterate and invoke
    def batch(
        self,
        inputs: list[Input],
        return_exceptions: bool = False,
        max_concurrency: int | None = None,
    ) -> list[Output | BaseException]:
        """Default synchronous batch processing by iterating invokes.

        Subclasses may override this for more efficient batch processing.

        Args:
            inputs: A list of inputs.

        Returns:
            A list of outputs.

        """
        def _invoke(input_item: Input) -> Output | BaseException:
            try:
                return self.invoke(input_item)

            except BaseException as e:
                if not return_exceptions:
                    raise e
                return e

        return [_invoke(input) for input in inputs]

    # Default abatch implementation: gather ainvokes
    async def abatch(
        self,
        inputs: list[Input],
        return_exceptions: bool = False,
        max_concurrency: int | None = None,
    ) -> list[Output | BaseException]:
        """Default asynchronous batch processing using asyncio.gather.

        Subclasses may override this for more efficient batch processing.
        Uses an asyncio.Semaphore to limit concurrency if max_concurrency is set.

        Args:
            inputs: A list of inputs.
            return_exceptions: If True, return exceptions instead of raising them.
            max_concurrency: Maximum concurrent invocations. Defaults to None (no limit).

        Returns:
            A list of outputs, potentially including BaseException instances if
            return_exceptions is True.

        """
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be None or a positive integer")

        return [
            result
            for current_batch in batch(inputs, max_concurrency)
            for result in await asyncio.gather(
                *[self.ainvoke(input_item) for input_item in current_batch],
                return_exceptions=return_exceptions
            )
        ]

    # Default stream implementation: invoke and yield single result
    # def stream(self, input: Input) -> Iterable[Output]:
    #     ...

    # Default astream implementation: ainvoke and yield single result
    # def astream(self, input: Input) -> AsyncIterable[Output]:
    #     ...


@dataclass(frozen=True)
class RunnableSequence[Input, Output](RunnableBase[Input, Output]):
    """Represents a sequence of two Runnables chained together (first | second).

    A RunnableSequence executes its first runnable, takes its output, and passes
    it as input to the second runnable. It inherits the `__or__` method from
    `RunnableBase`, allowing further chaining like `(runnable1 | runnable2) | runnable3`.

    Attributes:
        first: The first runnable in the sequence (`Runnable[Input, Intermediate]`).
        second: The second runnable in the sequence (`Runnable[Intermediate, Output]`).

    """

    first: Runnable[Input, Any]
    second: Runnable[Any, Output]

    def invoke(self, input: Input) -> Output:
        """Invokes the sequence synchronously: `second.invoke(first.invoke(input))`.

        Args:
            input: The initial input to the first runnable.

        Returns:
            The final output from the second runnable.

        """
        intermediate_result = self.first.invoke(input)
        return self.second.invoke(intermediate_result)

    async def ainvoke(self, input: Input) -> Output:
        """Invokes the sequence asynchronously: `await second.ainvoke(await first.ainvoke(input))`.

        Args:
            input: The initial input to the first runnable.

        Returns:
            An awaitable resolving to the final output from the second runnable.

        """
        intermediate_result = await self.first.ainvoke(input)
        return await self.second.ainvoke(intermediate_result)

    # Sequence stream: stream first, then stream second for each chunk
    # def stream(self, input: Input) -> Iterable[Output]:
    #     ...

    # Sequence astream: astream first, then astream second for each chunk
    # def astream(self, input: Input) -> AsyncIterable[Output]:
    #     ...

@dataclass(frozen=True)
class RunnableLambda[Input, Output](RunnableBase[Input, Output]):
    """Wraps a Python function or coroutine function to make it a Runnable.

    This allows integrating arbitrary Python callables into a Runnable sequence.
    It handles both synchronous and asynchronous functions appropriately.

    Attributes:
        _func: The wrapped callable (sync or async function).
        _is_async: Boolean flag indicating if the wrapped function is async.

    """

    _func: WrappableFunc[Input, Output]

    @property
    def _is_async(self) -> bool:
        return inspect.iscoroutinefunction(self._func)

    def invoke(self, input: Input) -> Output:
        """Invokes the wrapped function synchronously.

        Args:
            input: The input passed to the wrapped function.

        Returns:
            The output of the wrapped function.

        Raises:
            TypeError: If the wrapped function is asynchronous (use `ainvoke`).

        """
        if self._is_async:
            func_name = getattr(self._func, "__name__", repr(self._func))
            raise TypeError(
                f"Cannot synchronously `invoke` coroutine function {func_name}. "
                "Use `ainvoke` instead."
            )
        return self._func(input)  # type: ignore

    async def ainvoke(self, input: Input) -> Output:
        """Invokes the wrapped function or coroutine asynchronously.

        If the wrapped function is synchronous, it runs it in a thread pool executor
        to avoid blocking the event loop.

        Args:
            input: The input passed to the wrapped function/coroutine.

        Returns:
            An awaitable resolving to the output of the wrapped function/coroutine.

        """
        return (
            await self._func(input)  # type: ignore
            if self._is_async
            else await asyncio.to_thread(self._func, input)  # type: ignore[arg-type]
        )


def runnable[Input, Output](
    func: WrappableFunc[Input, Output],
) -> RunnableLambda[Input, Output]:
    """Decorator to easily wrap a Python callable into a RunnableLambda.

    Example:
        @runnable
        def add_one(x: int) -> int:
            return x + 1

        chain = add_one | add_one
        result = chain.invoke(5) # result will be 7

    Args:
        func: The function or coroutine function to wrap.

    Returns:
        A RunnableLambda instance wrapping the function.

    """
    return RunnableLambda[Input, Output](func)
