import sys
import importlib

import pytest
from PIL import Image

from tinylcel.messages.chunks import HumanMessageChunk


def test_image_chunk_from_pil_image(tmp_path):
    from tinylcel.messages.image_chunk import ImageChunk

    # Create a small red image
    img = Image.new('RGB', (2, 2), color=(255, 0, 0))
    chunk = ImageChunk(img)
    assert isinstance(chunk, HumanMessageChunk)
    assert chunk.role == 'human'
    content = chunk.content

    assert isinstance(chunk.content, list)
    assert len(chunk.content) == 1
    content = chunk.content[0]

    assert isinstance(content, dict)
    assert content.get('type') == 'image_url'
    image_url = content['image_url']
    assert isinstance(image_url, dict)
    url = image_url.get('url')
    assert isinstance(url, str)
    assert url.startswith('data:image/webp;base64,')
    detail = image_url.get('detail')
    assert detail == 'auto'


def test_image_chunk_from_path(tmp_path):
    from tinylcel.messages.image_chunk import ImageChunk

    img_path = tmp_path / 'test.png'
    img = Image.new('RGB', (1, 1), color=(0, 255, 0))
    img.save(img_path, format='PNG')
    chunk = ImageChunk(img_path)
    content = chunk.content[0]
    url = content['image_url']['url']
    # Should detect correct mime for webp default
    assert url.startswith('data:image/webp;base64,')

@pytest.mark.parametrize('fmt, prefix', [
    ('jpeg', 'data:image/jpeg;base64,'),
    ('png', 'data:image/png;base64,'),
])
def test_image_chunk_different_formats(tmp_path, fmt, prefix):
    from tinylcel.messages.image_chunk import ImageChunk
    img_path = tmp_path / f'test.{fmt}'
    img = Image.new('RGB', (1, 1), color=(0, 0, 255))
    img.save(img_path, format=fmt.upper())
    chunk = ImageChunk(img_path, wire_format=fmt)
    content = chunk.content[0]
    url = content['image_url']['url']
    assert url.startswith(prefix) 


@pytest.mark.parametrize('missing_module', ['PIL', 'magic'])
def test_missing_pillow_dependency(missing_module, monkeypatch):
    MODULE = 'tinylcel.messages.image_chunk'

    # Simulate missing Pillow by removing PIL modules
    monkeypatch.setitem(sys.modules, missing_module, None)

    # Ensure module is not cached
    if MODULE in sys.modules:
        del sys.modules[MODULE]
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(MODULE)
    
    msg = str(excinfo.value)
    assert 'please install' in msg.lower()
    assert 'python-magic' in msg
    assert 'pillow' in msg 