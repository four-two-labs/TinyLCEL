import time
import asyncio
from typing import Any
from typing import TypeVar
from typing import Callable
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from tinylcel.runnable import runnable
from tinylcel.runnable import RunnableBase
from tinylcel.runnable import RunnableLambda
from tinylcel.runnable import RunnableSequence

# --- Helper Functions and Classes ---

# TypeVars for generic MockRunnable, inherited from RunnableBase
Input = TypeVar('Input')
Output = TypeVar('Output')


def sync_add_one(x: int) -> int:
    if not isinstance(x, int):
        raise TypeError('Input must be an integer')
    return x + 1


async def async_add_one(x: int) -> int:
    if not isinstance(x, int):
        raise TypeError('Input must be an integer')
    await asyncio.sleep(0.01)  # Simulate async work
    return x + 1


async def async_identity(x: Any) -> Any:
    await asyncio.sleep(0.01)
    return x


def sync_raises(_x: Any) -> None:
    raise ValueError('Sync error')


async def async_raises(_x: Any) -> None:
    await asyncio.sleep(0.01)
    raise ValueError('Async error')


# Concrete runnable for testing base methods if needed
class MockRunnable[Input, Output](RunnableBase[Input, Output]):  # noqa: D101
    def __init__(self, invoke_mock: Mock | None = None, ainvoke_mock: AsyncMock | None = None):  # noqa: D107
        self._invoke: Mock = invoke_mock or Mock(side_effect=lambda x: x + 1)
        self._ainvoke: AsyncMock = ainvoke_mock or AsyncMock(side_effect=async_identity)

    def invoke(self, input_val: Input) -> Output:  # noqa: D102
        return self._invoke(input_val)  # type: ignore[no-any-return]

    async def ainvoke(self, input_val: Input) -> Output:  # noqa: D102
        return await self._ainvoke(input_val)  # type: ignore[no-any-return]


# --- Tests ---

# RunnableBase Tests


def _original_multiply_by_two(x: int) -> int:
    return x * 2


async def _original_async_multiply_by_two(x: int) -> int:
    await asyncio.sleep(0)
    return x * 2


def test_runnable_base_or_operator() -> None:
    """Test the | operator creates a RunnableSequence."""
    r1: MockRunnable[int, int] = MockRunnable[int, int]()
    r2: MockRunnable[int, int] = MockRunnable[int, int]()
    sequence = r1 | r2
    assert isinstance(sequence, RunnableSequence)
    assert sequence.first is r1
    assert sequence.second is r2


def test_runnable_base_or_operator_type_error() -> None:
    """Test the | operator raises TypeError for non-Runnables."""
    r1: MockRunnable[Any, Any] = MockRunnable[Any, Any]()
    with pytest.raises(TypeError):
        _ = r1 | (lambda x: x)  # type: ignore


@pytest.mark.parametrize(('return_exceptions', 'expect_raise'), [(False, True), (True, False)])
def test_runnable_base_batch_default(return_exceptions: bool, expect_raise: bool) -> None:
    """Test the default batch implementation iterates invoke."""
    mock_invoke = Mock()
    mock_invoke.side_effect = [1, ValueError('error'), 3]
    r: MockRunnable[Any, Any] = MockRunnable[Any, Any](invoke_mock=mock_invoke)
    inputs = [0, 'a', 2]

    if expect_raise:
        with pytest.raises(ValueError, match='error'):
            r.batch(inputs, return_exceptions=return_exceptions)
        assert mock_invoke.call_count == 2
        mock_invoke.assert_has_calls([call(0), call('a')])
    else:
        results = r.batch(inputs, return_exceptions=return_exceptions)
        assert mock_invoke.call_count == 3
        mock_invoke.assert_has_calls([call(0), call('a'), call(2)])
        assert results[0] == 1
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == 'error'
        assert results[2] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(('return_exceptions', 'expect_raise'), [(False, True), (True, False)])
async def test_runnable_base_abatch_default(return_exceptions: bool, expect_raise: bool) -> None:
    """Test the default abatch implementation uses gather."""
    mock_ainvoke = AsyncMock()
    mock_ainvoke.side_effect = [1, ValueError('async error'), 3]
    r: MockRunnable[Any, Any] = MockRunnable[Any, Any](ainvoke_mock=mock_ainvoke)
    inputs = [0, 'a', 2]

    if expect_raise:
        with pytest.raises(ValueError, match='async error'):
            await r.abatch(inputs, return_exceptions=return_exceptions)
        # gather calls all, even if one fails early
        assert mock_ainvoke.await_count == 3
        mock_ainvoke.assert_has_awaits([call(0), call('a'), call(2)])
    else:
        results = await r.abatch(inputs, return_exceptions=return_exceptions)
        assert mock_ainvoke.await_count == 3
        mock_ainvoke.assert_has_awaits([call(0), call('a'), call(2)])
        assert results[0] == 1
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == 'async error'
        assert results[2] == 3


@pytest.mark.asyncio
async def test_runnable_base_abatch_concurrency() -> None:
    """Test abatch respects max_concurrency."""
    sleep_time = 0.1
    call_log = []

    async def slow_ainvoke(x: int) -> int:
        call_log.append(f'start {x}')
        await asyncio.sleep(sleep_time)
        call_log.append(f'end {x}')
        return x + 1

    r: MockRunnable[int, int] = MockRunnable[int, int](ainvoke_mock=AsyncMock(side_effect=slow_ainvoke))
    inputs = [1, 2, 3, 4, 5]
    max_concurrency = 2

    start_time = time.monotonic()
    results = await r.abatch(inputs, max_concurrency=max_concurrency)
    end_time = time.monotonic()

    assert results == [2, 3, 4, 5, 6]
    # Expected time: roughly ceil(len(inputs) / max_concurrency) * sleep_time
    # ceil(5/2) * 0.1 = 3 * 0.1 = 0.3
    # Allow some buffer
    expected_min_time = 3 * sleep_time
    expected_max_time = 3.5 * sleep_time
    actual_time = end_time - start_time
    # Check call log pattern (e.g., starts should be grouped by concurrency)
    # Example: start 1, start 2, end 1, start 3, end 2, start 4, end 3, start 5, end 4, end 5
    assert actual_time >= expected_min_time, f'Expected min time {expected_min_time:.2f}s, got {actual_time:.2f}s'
    assert actual_time < expected_max_time, f'Expected max time {expected_max_time:.2f}s, got {actual_time:.2f}s'


@pytest.mark.asyncio
async def test_runnable_base_abatch_invalid_concurrency() -> None:
    """Test abatch raises ValueError for invalid max_concurrency."""
    r: MockRunnable[Any, Any] = MockRunnable[Any, Any]()
    with pytest.raises(ValueError, match='max_concurrency must be None or a positive integer'):
        await r.abatch([1, 2], max_concurrency=0)
    with pytest.raises(ValueError, match='max_concurrency must be None or a positive integer'):
        await r.abatch([1, 2], max_concurrency=-1)


# RunnableLambda Tests


def test_runnable_lambda_sync_invoke() -> None:
    r: RunnableLambda[int, int] = RunnableLambda(sync_add_one)
    assert r.invoke(1) == 2
    with pytest.raises(TypeError):
        r.invoke('a')  # type: ignore[arg-type]


@pytest.mark.asyncio
@patch('asyncio.to_thread')
async def test_runnable_lambda_sync_ainvoke(mock_to_thread: MagicMock) -> None:
    mock_to_thread.return_value = 2  # Mock the result of running in thread
    r: RunnableLambda[int, int] = RunnableLambda(sync_add_one)
    assert await r.ainvoke(1) == 2
    mock_to_thread.assert_awaited_once_with(sync_add_one, 1)

    # Test error propagation from thread
    mock_to_thread.side_effect = TypeError('Input must be an integer')
    with pytest.raises(TypeError):
        await r.ainvoke('a')  # type: ignore[arg-type]
    mock_to_thread.assert_awaited_with(sync_add_one, 'a')


@pytest.mark.asyncio
async def test_runnable_lambda_async_ainvoke() -> None:
    r: RunnableLambda[int, int] = RunnableLambda(async_add_one)
    assert await r.ainvoke(1) == 2
    with pytest.raises(TypeError):
        await r.ainvoke('a')  # type: ignore[arg-type]


def test_runnable_lambda_async_invoke_raises() -> None:
    r: RunnableLambda[int, int] = RunnableLambda(async_add_one)
    with pytest.raises(TypeError):
        r.invoke(1)  # type: ignore[arg-type]


@pytest.mark.parametrize('return_exceptions', [False, True])
def test_runnable_lambda_sync_batch(return_exceptions: bool) -> None:
    r: RunnableLambda[int, int] = RunnableLambda(sync_add_one)
    inputs = [1, 2, 'a', 4]
    if return_exceptions:
        results = r.batch(inputs, return_exceptions=True)  # type: ignore[arg-type]
        assert results[0] == 2
        assert results[1] == 3
        assert isinstance(results[2], TypeError)
        assert results[3] == 5
    else:
        with pytest.raises(TypeError):
            r.batch(inputs, return_exceptions=False)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize('return_exceptions', [False, True])
async def test_runnable_lambda_async_abatch(return_exceptions: bool) -> None:
    r: RunnableLambda[int, int] = RunnableLambda(async_add_one)
    inputs = [1, 2, 'a', 4]
    if return_exceptions:
        results = await r.abatch(inputs, return_exceptions=True)  # type: ignore[arg-type]
        assert results[0] == 2
        assert results[1] == 3
        assert isinstance(results[2], TypeError)
        assert results[3] == 5
    else:
        with pytest.raises(TypeError):
            await r.abatch(inputs, return_exceptions=False)  # type: ignore[arg-type]


def test_runnable_lambda_async_batch_raises() -> None:
    """Verify sync batch raises TypeError for async lambda."""
    r: RunnableLambda[int, int] = RunnableLambda(async_add_one)
    with pytest.raises(TypeError):
        r.batch([1, 2])  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize('return_exceptions', [False, True])
@patch('asyncio.to_thread')  # Since default abatch uses ainvoke which uses to_thread
async def test_runnable_lambda_sync_abatch(mock_to_thread: MagicMock, return_exceptions: bool) -> None:
    r: RunnableLambda[int, int] = RunnableLambda(sync_add_one)
    inputs = [1, 2, 'a', 4]

    # Define side effect for to_thread based on input
    async def side_effect(func: Callable[[Any], Any], arg: Any) -> Any:
        # Need to actually call the function to check for errors
        return func(arg)

    mock_to_thread.side_effect = side_effect

    if return_exceptions:
        results = await r.abatch(inputs, return_exceptions=True)  # type: ignore[arg-type]
        # Despite mocking, the structure should work
        assert results[0] == 2
        assert results[1] == 3
        assert isinstance(results[2], TypeError)
        assert results[3] == 5
        assert mock_to_thread.await_count == 4
        mock_to_thread.assert_has_awaits(
            [
                call(sync_add_one, 1),
                call(sync_add_one, 2),
                call(sync_add_one, 'a'),
                call(sync_add_one, 4),
            ],
            any_order=True,
        )
    else:
        with pytest.raises(TypeError):
            await r.abatch(inputs, return_exceptions=False)  # type: ignore[arg-type]
        # gather might await all even if one fails
        assert mock_to_thread.await_count == 4


# runnable Decorator Tests


def test_runnable_decorator_sync() -> None:
    @runnable
    def decorated_add_one(x: int) -> int:
        return x + 1

    decorated_runnable: RunnableLambda[int, int] = decorated_add_one
    assert isinstance(decorated_runnable, RunnableLambda)
    assert decorated_runnable.invoke(5) == 6


@pytest.mark.asyncio
async def test_runnable_decorator_async() -> None:
    @runnable  # type: ignore[arg-type]
    async def decorated_async_add_one(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    # Explicitly annotate the resulting runnable type
    decorated_runnable: RunnableLambda[int, int] = decorated_async_add_one
    assert isinstance(decorated_runnable, RunnableLambda)
    assert await decorated_runnable.ainvoke(5) == 6
    with pytest.raises(TypeError):
        decorated_runnable.invoke(5)


# RunnableSequence Tests


@runnable
def multiply_by_two(x: int) -> int:
    return x * 2


@runnable  # type: ignore[arg-type]
async def async_multiply_by_two(x: int) -> int:
    await asyncio.sleep(0)
    return x * 2


r_sync_add: RunnableLambda[int, int] = RunnableLambda(sync_add_one)
r_async_add: RunnableLambda[int, int] = RunnableLambda(async_add_one)
r_sync_mul: RunnableLambda[int, int] = RunnableLambda(_original_multiply_by_two)
r_async_mul: RunnableLambda[int, int] = RunnableLambda(_original_async_multiply_by_two)


def test_runnable_sequence_invoke_sync_sync() -> None:
    seq = r_sync_add | r_sync_mul
    assert seq.invoke(5) == 12  # (5+1)*2


def test_runnable_sequence_invoke_sync_async_raises() -> None:
    seq = r_sync_add | r_async_mul
    with pytest.raises(TypeError):
        seq.invoke(5)


def test_runnable_sequence_invoke_async_sync_raises() -> None:
    seq = r_async_add | r_sync_mul
    with pytest.raises(TypeError):
        seq.invoke(5)


@pytest.mark.asyncio
async def test_runnable_sequence_ainvoke_sync_sync() -> None:
    seq = r_sync_add | r_sync_mul
    assert await seq.ainvoke(5) == 12  # (5+1)*2


@pytest.mark.asyncio
async def test_runnable_sequence_ainvoke_sync_async() -> None:
    seq = r_sync_add | r_async_mul
    assert await seq.ainvoke(5) == 12  # (5+1)*2


@pytest.mark.asyncio
async def test_runnable_sequence_ainvoke_async_sync() -> None:
    seq = r_async_add | r_sync_mul
    assert await seq.ainvoke(5) == 12  # (5+1)*2


@pytest.mark.asyncio
async def test_runnable_sequence_ainvoke_async_async() -> None:
    seq = r_async_add | r_async_mul
    assert await seq.ainvoke(5) == 12  # (5+1)*2


def test_runnable_sequence_chaining() -> None:
    seq = r_sync_add | r_sync_mul | r_sync_add  # type: ignore[operator]
    assert isinstance(seq, RunnableSequence)
    assert isinstance(seq.first, RunnableSequence)  # This part of the chain is a sequence
    # Check structure of the (r_sync_add | r_sync_mul) part
    assert isinstance(seq.first.first, RunnableLambda)
    assert seq.first.first._func is sync_add_one
    assert isinstance(seq.first.second, RunnableLambda)
    assert seq.first.second._func is _original_multiply_by_two
    # Check the last part of the chain
    assert isinstance(seq.second, RunnableLambda)
    assert seq.second._func is sync_add_one
    assert seq.invoke(5) == 13


def test_runnable_sequence_invoke_error() -> None:
    r_sync_err: RunnableLambda[Any, None] = RunnableLambda(sync_raises)
    seq1 = r_sync_add | r_sync_err
    seq2 = r_sync_err | r_sync_add  # type: ignore[operator]

    with pytest.raises(ValueError, match='Sync error'):
        seq1.invoke(1)
    with pytest.raises(ValueError, match='Sync error'):
        seq2.invoke(1)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_runnable_sequence_ainvoke_error() -> None:
    r_sync_err: RunnableLambda[Any, None] = RunnableLambda(sync_raises)
    r_async_err: RunnableLambda[Any, None] = RunnableLambda(async_raises)

    seq1 = r_sync_add | r_sync_err
    seq2 = r_sync_add | r_async_err
    seq3 = r_async_add | r_sync_err
    seq4 = r_async_add | r_async_err
    seq5 = r_sync_err | r_sync_add  # type: ignore[operator]
    seq6 = r_async_err | r_sync_add  # type: ignore[operator]

    with pytest.raises(ValueError, match='error'):
        await seq1.ainvoke(1)
    with pytest.raises(ValueError, match='error'):
        await seq2.ainvoke(1)
    with pytest.raises(ValueError, match='error'):
        await seq3.ainvoke(1)
    with pytest.raises(ValueError, match='error'):
        await seq4.ainvoke(1)
    with pytest.raises(ValueError, match='error'):
        await seq5.ainvoke(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='error'):
        await seq6.ainvoke(1)  # type: ignore[arg-type]


@pytest.mark.parametrize('return_exceptions', [False, True])
def test_runnable_sequence_batch(return_exceptions: bool) -> None:
    seq = r_sync_add | r_sync_mul
    inputs = [1, 2, 'a', 4]  # 'a' will cause TypeError in add
    if return_exceptions:
        results = seq.batch(inputs, return_exceptions=True)  # type: ignore[arg-type]
        assert results[0] == 4  # (1+1)*2
        assert results[1] == 6  # (2+1)*2
        assert isinstance(results[2], TypeError)  # Error from first step
        assert results[3] == 10  # (4+1)*2
    else:
        with pytest.raises(TypeError):
            seq.batch(inputs, return_exceptions=False)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize('return_exceptions', [False, True])
async def test_runnable_sequence_abatch(return_exceptions: bool) -> None:
    seq_aa = r_async_add | r_async_mul
    seq_sa = r_sync_add | r_async_mul
    seq_as = r_async_add | r_sync_mul

    inputs = [1, 2, 'a', 4]  # 'a' will cause TypeError

    for seq in [seq_aa, seq_sa, seq_as]:
        if return_exceptions:
            results = await seq.abatch(inputs, return_exceptions=True)  # type: ignore[arg-type]
            assert results[0] == 4  # (1+1)*2
            assert results[1] == 6  # (2+1)*2
            assert isinstance(results[2], TypeError)  # Error from first step
            assert results[3] == 10  # (4+1)*2
        else:
            with pytest.raises(TypeError):
                await seq.abatch(inputs, return_exceptions=False)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_runnable_sequence_abatch_concurrency() -> None:
    """Test sequence abatch respects max_concurrency."""
    sleep_time = 0.1
    call_log = []

    @runnable  # type: ignore[arg-type]
    async def slow_step1(x: int) -> int:
        call_log.append(f's1 start {x}')
        await asyncio.sleep(sleep_time / 2)
        call_log.append(f's1 end {x}')
        return x + 1

    slow_step1_runnable: RunnableLambda[int, int] = slow_step1

    @runnable  # type: ignore[arg-type]
    async def slow_step2(x: int) -> int:
        call_log.append(f's2 start {x}')
        await asyncio.sleep(sleep_time / 2)
        call_log.append(f's2 end {x}')
        return x * 2

    slow_step2_runnable: RunnableLambda[int, int] = slow_step2

    seq = slow_step1_runnable | slow_step2_runnable
    inputs = [1, 2, 3, 4, 5]
    max_concurrency = 2

    start_time = time.monotonic()
    results = await seq.abatch(inputs, max_concurrency=max_concurrency)
    end_time = time.monotonic()

    assert results == [4, 6, 8, 10, 12]  # ((x+1)*2)
    # Expected time: roughly ceil(len(inputs) / max_concurrency) * total_sleep_per_item
    # ceil(5/2) * (0.1/2 + 0.1/2) = 3 * 0.1 = 0.3
    expected_min_time = 3 * sleep_time
    expected_max_time = 3.5 * sleep_time  # Allow buffer
    actual_time = end_time - start_time

    assert actual_time >= expected_min_time, f'Expected min time {expected_min_time:.2f}s, got {actual_time:.2f}s'
    assert actual_time < expected_max_time, f'Expected max time {expected_max_time:.2f}s, got {actual_time:.2f}s'
