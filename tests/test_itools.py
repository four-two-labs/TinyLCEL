import asyncio
from typing import Any
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Optional
from typing import AsyncIterable

import pytest

from tinylcel.itools import azip
from tinylcel.itools import take
from tinylcel.itools import atake
from tinylcel.itools import batch
from tinylcel.itools import abatch
from tinylcel.itools import arange


# Helper async generator for testing
async def async_gen(data: List[Any]) -> AsyncIterable[Any]:
    for item in data:
        await asyncio.sleep(0)  # Yield control
        yield item


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'args, expected',
    [
        ((5,), [0, 1, 2, 3, 4]),
        ((2, 5), [2, 3, 4]),
        ((1, 6, 2), [1, 3, 5]),
        ((0,), []),
        ((5, 2), []),
        ((5, 5), []),
    ],
)
async def test_arange(args: Tuple[int, ...], expected: List[int]) -> None:
    """Test the arange async generator."""
    result = [i async for i in arange(*args)]  # type: ignore[attr-defined]
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'a_data, b_data, expected',
    [
        ([1, 2, 3], ['a', 'b', 'c'], [(1, 'a'), (2, 'b'), (3, 'c')]),
        ([1, 2], ['a', 'b', 'c'], [(1, 'a'), (2, 'b')]),
        ([1, 2, 3], ['a', 'b'], [(1, 'a'), (2, 'b')]),
        ([], ['a', 'b'], []),
        ([1, 2], [], []),
        ([], [], []),
    ],
)
async def test_azip(a_data: List[Any], b_data: List[Any], expected: List[Tuple[Any, Any]]) -> None:
    """Test the azip async generator."""
    a = async_gen(a_data)
    b = async_gen(b_data)
    result = [item async for item in azip(a, b)]
    assert result == expected


@pytest.mark.parametrize(
    'iterable, n, expected',
    [
        ([1, 2, 3, 4, 5], 3, [1, 2, 3]),
        (range(5), 3, [0, 1, 2]),
        ([1, 2, 3], 5, [1, 2, 3]),
        ([1, 2, 3], 0, []),
        ([1, 2, 3], None, [1, 2, 3]),
        ([], 3, []),
        ([], None, []),
    ],
)
def test_take(iterable: Iterable[Any], n: Optional[int], expected: List[Any]) -> None:
    """Test the take function."""
    result = list(take(iterable, n))
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'data, n, expected',
    [
        ([1, 2, 3, 4, 5], 3, [1, 2, 3]),
        ([1, 2, 3], 5, [1, 2, 3]),
        ([1, 2, 3], 0, []),
        ([1, 2, 3], None, [1, 2, 3]),
        ([], 3, []),
        ([], None, []),
    ],
)
async def test_atake(data: List[Any], n: Optional[int], expected: List[Any]) -> None:
    """Test the atake async generator."""
    aiterable = async_gen(data)
    result = [item async for item in atake(aiterable, n)]
    assert result == expected


@pytest.mark.parametrize(
    'iterable, batch_size, expected',
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 2], [3, 4], [5, 6]]),
        (range(6), 2, [[0, 1], [2, 3], [4, 5]]),
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
        ([1, 2, 3, 4, 5], 1, [[1], [2], [3], [4], [5]]),
        ([1, 2, 3], 5, [[1, 2, 3]]),
        ([1, 2, 3], None, [[1, 2, 3]]),
        ([], 2, []),
        ([], None, []),  # Corrected: batch(None) on empty yields nothing
    ],
)
def test_batch(iterable: Iterable[Any], batch_size: Optional[int], expected: List[List[Any]]) -> None:
    """Test the batch function."""
    result = list(batch(iterable, batch_size))
    assert result == expected  # Simplified assertion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'data, batch_size, expected',
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 2], [3, 4], [5, 6]]),
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
        ([1, 2, 3, 4, 5], 1, [[1], [2], [3], [4], [5]]),
        ([1, 2, 3], 5, [[1, 2, 3]]),
        ([1, 2, 3], None, [[1, 2, 3]]),
        ([], 2, []),
        ([], None, []),  # Implementation yields nothing for empty async iterable with None size
    ],
)
async def test_abatch(data: List[Any], batch_size: Optional[int], expected: List[List[Any]]) -> None:
    """Test the abatch async generator."""
    aiterable = async_gen(data)
    result = [item async for item in abatch(aiterable, batch_size)]
    assert result == expected
