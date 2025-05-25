import pytest

from tinylcel.messages.chunks import AIMessageChunk
from tinylcel.messages.chunks import BaseMessageChunk
from tinylcel.messages.chunks import HumanMessageChunk
from tinylcel.messages.chunks import SystemMessageChunk


@pytest.fixture
def simple_block() -> dict[str, str]:
    return {'type': 'text', 'text': 'hello'}


def test_human_chunk_add_simple_blocks(simple_block: dict[str, str]) -> None:
    c1 = HumanMessageChunk(content=simple_block)
    c2 = HumanMessageChunk(content=simple_block)
    result = c1 + c2
    assert isinstance(result, HumanMessageChunk)
    assert result.content == [simple_block, simple_block]


def test_human_chunk_add_list_and_dict(simple_block: dict[str, str]) -> None:
    c1 = HumanMessageChunk(content=[simple_block])
    c2 = HumanMessageChunk(content=simple_block)
    result = c1 + c2
    assert result.content == [simple_block, simple_block]
    c3 = HumanMessageChunk(content=simple_block)
    c4 = HumanMessageChunk(content=[simple_block])
    result2 = c3 + c4
    assert result2.content == [simple_block, simple_block]


def test_ai_chunk_add_and_role_preserved(simple_block: dict[str, str]) -> None:
    c1 = AIMessageChunk(content=simple_block)
    c2 = AIMessageChunk(content=simple_block)
    result = c1 + c2
    assert isinstance(result, AIMessageChunk)
    assert result.role == 'ai'


def test_system_chunk_cannot_add_human(simple_block: dict[str, str]) -> None:
    c1 = SystemMessageChunk(content=simple_block)
    c2 = HumanMessageChunk(content=simple_block)
    with pytest.raises(TypeError):
        _ = c1 + c2


def test_incompatible_content_raise() -> None:
    # Create a dummy chunk subclass to test unsupported content types
    class DummyChunk(BaseMessageChunk):
        role = 'human'

    # Use non-mapping content types to trigger TypeError
    d1 = DummyChunk(content=123)  # type: ignore[arg-type]
    d2 = DummyChunk(content=456)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _ = d1 + d2


def test_human_chunk_concat_lists(simple_block: dict[str, str]) -> None:
    c1 = HumanMessageChunk(content=[simple_block, simple_block])
    c2 = HumanMessageChunk(content=[simple_block])
    result = c1 + c2
    assert isinstance(result, HumanMessageChunk)
    assert result.content == [simple_block, simple_block, simple_block]


def test_system_chunk_add_and_concat(simple_block: dict[str, str]) -> None:
    # Simple dict + dict
    s1 = SystemMessageChunk(content=simple_block)
    s2 = SystemMessageChunk(content=simple_block)
    res1 = s1 + s2
    assert isinstance(res1, SystemMessageChunk)
    assert res1.content == [simple_block, simple_block]
    # List + list
    s3 = SystemMessageChunk(content=[simple_block])
    s4 = SystemMessageChunk(content=[simple_block, simple_block])
    res2 = s3 + s4
    assert res2.content == [simple_block, simple_block, simple_block]


@pytest.mark.parametrize('ChunkClass', [HumanMessageChunk, AIMessageChunk, SystemMessageChunk])
def test_subtype_preserved(
    ChunkClass: type[BaseMessageChunk],  # noqa: N803
    simple_block: dict[str, str],
) -> None:
    c1 = ChunkClass(content=simple_block)
    c2 = ChunkClass(content=simple_block)
    res = c1 + c2
    assert isinstance(res, ChunkClass)
    assert res.role == c1.role


def test_text_chunk_basic() -> None:
    from tinylcel.messages.chunks import text_chunk

    chunk = text_chunk('hello')
    from tinylcel.messages.chunks import HumanMessageChunk

    assert isinstance(chunk, HumanMessageChunk)
    assert chunk.role == 'human'
    assert isinstance(chunk.content, list)
    assert chunk.content == [{'type': 'text', 'text': 'hello'}]


def test_text_chunk_alias() -> None:
    from tinylcel.messages.chunks import TextChunk
    from tinylcel.messages.chunks import text_chunk

    chunk1 = text_chunk('hi')
    chunk2 = TextChunk('hi')
    assert isinstance(chunk2, type(chunk1))
    assert chunk2.content == chunk1.content


def test_text_chunk_concat() -> None:
    from tinylcel.messages.chunks import text_chunk

    c1 = text_chunk('foo')
    c2 = text_chunk('bar')
    res = c1 + c2
    assert isinstance(res, type(c1))
    assert res.content == [
        {'type': 'text', 'text': 'foo'},
        {'type': 'text', 'text': 'bar'},
    ]
