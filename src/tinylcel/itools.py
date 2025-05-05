import sys
from typing import Iterable
from typing import overload
from typing import AsyncIterable


@overload
async def arange(stop: int) -> AsyncIterable[int]: ...

@overload
async def arange(start: int, stop: int, step: int = 1) -> AsyncIterable[int]: ...

async def arange(*args: int) -> AsyncIterable[int]: # type: ignore[misc]
    """Asynchronously generate a range of integers.

    Args:
        start: The starting value of the range.
        stop: The stopping value of the range.
        step: The step value of the range.

    """
    for i in range(*args):
        yield i


async def azip[Ta, Tb](a: AsyncIterable[Ta], b: AsyncIterable[Tb]) -> AsyncIterable[tuple[Ta, Tb]]:
    """Zip two iterables together asynchronously.

    Args:
        a: The async iterable to zip with the iterable.
        b: The iterable to zip with the async iterable.

    """
    it1, it2 = aiter(a), aiter(b)
    while True:
        try:
            yield await anext(it1), await anext(it2)
        except StopAsyncIteration:
            break


def take[T](iterable: Iterable[T], n: int | None = None) -> Iterable[T]:
    """Take the first n items from an iterable.

    Args:
        iterable: The iterable to take items from.
        n: The number of items to take. If None, returns the entire iterable.

    Returns:
        An iterable yielding the first n items of the input iterable.

    """
    if n == 0:
        return
    effective_n = n or sys.maxsize
    for _, item in zip(range(effective_n), iterable, strict=False):
        yield item


async def atake[T](aiterable: AsyncIterable[T], n: int | None = None) -> AsyncIterable[T]:
    """Asynchronously take the first n items from an async iterable.

    Args:
        aiterable: The async iterable to take items from.
        n: The number of items to take. If None, returns the entire async iterable.

    Returns:
        An async iterable yielding the first n items of the input async iterable.

    """
    if n == 0:
        return
    effective_n = n or sys.maxsize
    async for _, item in azip(arange(effective_n), aiterable): # type: ignore[arg-type]
        yield item

def batch[T](iterable: Iterable[T], batch_size: int | None = None) -> Iterable[list[T]]:
    """Batch items from an iterable into lists of a specified size.

    Args:
        iterable: The iterable to batch items from.
        batch_size: The size of each batch. If None, returns the entire iterable as a single batch.

    Returns:
        An iterable yielding lists of items from the input iterable, each of the specified batch size.

    """
    it = iter(iterable)
    while (batch := list(take(it, batch_size))):
        yield batch

async def abatch[T](aiterable: AsyncIterable[T], batch_size: int | None = None) -> AsyncIterable[list[T]]:
    """Asynchronously batch items from an async iterable into lists of a specified size.

    Args:
        aiter: The async iterable to batch items from.
        batch_size: The size of each batch. If None, returns the entire async iterable as a single batch.

    Returns:
        An async iterable yielding lists of items from the input async iterable, each of the specified batch size.

    """
    it = aiter(aiterable)
    while (batch := [e async for e in atake(it, batch_size)]):
        yield batch
