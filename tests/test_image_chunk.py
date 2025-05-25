import sys
import importlib
from typing import cast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image  # type: ignore[import-not-found]

from tinylcel.messages.chunks import HumanMessageChunk

if TYPE_CHECKING:
    from tinylcel.messages.base import MessageContentBlock


def test_image_chunk_from_pil_image(tmp_path: Path) -> None:
    from tinylcel.messages.image_chunk import ImageChunk

    # Create a small red image
    img = Image.new('RGB', (2, 2), color=(255, 0, 0))
    # The ImageChunk needs a file path to create a data URL,
    # so we save the PIL image to a temporary file.
    img_path = tmp_path / 'test_pil_image.png'
    img.save(img_path, format='PNG')

    chunk = ImageChunk(img_path)
    assert isinstance(chunk, HumanMessageChunk)
    assert chunk.role == 'human'

    # Type narrowing: ensure content is a list
    content = chunk.content
    assert isinstance(content, list)
    assert len(content) == 1
    content_list = cast('list[MessageContentBlock]', content)
    content_item = content_list[0]

    assert isinstance(content_item, dict)
    content_dict = cast('dict[str, str | dict[str, str]]', content_item)
    assert content_dict.get('type') == 'image_url'
    image_url_value = content_dict['image_url']
    assert isinstance(image_url_value, dict)
    image_url_dict = cast('dict[str, str]', image_url_value)
    url = image_url_dict.get('url')
    assert isinstance(url, str)
    assert url.startswith('data:image/webp;base64,')  # Default format is webp
    detail = image_url_dict.get('detail')
    assert detail == 'auto'


def test_image_chunk_from_path(tmp_path: Path) -> None:
    from tinylcel.messages.image_chunk import ImageChunk

    img_path = tmp_path / 'test.png'
    img = Image.new('RGB', (1, 1), color=(0, 255, 0))
    img.save(img_path, format='PNG')
    chunk = ImageChunk(img_path)

    # Type narrowing: ensure content is a list
    content = chunk.content
    assert isinstance(content, list)
    content_list = cast('list[MessageContentBlock]', content)
    content_item = content_list[0]
    assert isinstance(content_item, dict)
    content_dict = cast('dict[str, str | dict[str, str]]', content_item)
    image_url_value = content_dict['image_url']
    assert isinstance(image_url_value, dict)
    image_url_dict = cast('dict[str, str]', image_url_value)
    url = image_url_dict.get('url')
    assert isinstance(url, str)
    # Should detect correct mime for webp default
    assert url.startswith('data:image/webp;base64,')


@pytest.mark.parametrize(
    ('fmt', 'prefix'),
    [
        ('jpeg', 'data:image/jpeg;base64,'),
        ('png', 'data:image/png;base64,'),
    ],
)
def test_image_chunk_different_formats(tmp_path: Path, fmt: str, prefix: str) -> None:
    from tinylcel.messages.image_chunk import ImageChunk

    img_path = tmp_path / f'test.{fmt}'
    img = Image.new('RGB', (1, 1), color=(0, 0, 255))
    img.save(img_path, format=fmt.upper())
    chunk = ImageChunk(img_path, wire_format=fmt)

    # Type narrowing: ensure content is a list
    content = chunk.content
    assert isinstance(content, list)
    content_list = cast('list[MessageContentBlock]', content)
    content_item = content_list[0]
    assert isinstance(content_item, dict)
    content_dict = cast('dict[str, str | dict[str, str]]', content_item)
    image_url_value = content_dict['image_url']
    assert isinstance(image_url_value, dict)
    image_url_dict = cast('dict[str, str]', image_url_value)
    url = image_url_dict.get('url')
    assert isinstance(url, str)
    assert url.startswith(prefix)


@pytest.mark.parametrize('missing_module', ['PIL', 'magic'])
def test_missing_pillow_dependency(missing_module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = 'tinylcel.messages.image_chunk'

    # Simulate missing Pillow by removing PIL modules
    monkeypatch.setitem(sys.modules, missing_module, None)

    # Ensure module is not cached
    if module in sys.modules:
        del sys.modules[module]
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(module)

    msg = str(excinfo.value)
    assert 'please install' in msg.lower()
    assert 'python-magic' in msg
    assert 'pillow' in msg
