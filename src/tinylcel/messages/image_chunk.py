#!/usr/bin/env python3
"""
ImageChunk: a HumanMessageChunk subclass that wraps an image as a base64 data-URL block.
"""
import io
import base64
import mimetypes
from pathlib import Path
from typing import Union

# Guard for required dependencies
try:
    import magic  # type: ignore
    _ = magic  # use to satisfy linter
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "The 'python-magic' and 'pillow' libraries are required to use ImageChunk. "
        "Please install them with 'pip install python-magic pillow'."
    ) from e

from tinylcel.messages.base import MessageContentBlock
from tinylcel.messages.chunks import HumanMessageChunk


class ImageChunk(HumanMessageChunk):
    """A message chunk representing an image encoded as a base64 data URL."""
    def __init__(
        self,
        image: Union[Path, str, Image.Image],
        wire_format: str = 'webp',
        quality: int = 85,
    ) -> None:
        # Load image
        img = image if isinstance(image, Image.Image) else Image.open(image)  # type: ignore
        # Encode image to bytes buffer
        buf = io.BytesIO()
        img.save(buf, format=wire_format.upper(), quality=quality)
        data = buf.getvalue()
        # Determine MIME type
        mime = mimetypes.guess_type(f'file.{wire_format.lower()}')[0] or f'image/{wire_format.lower()}'
        # Base64 encode
        b64 = base64.b64encode(data).decode('utf-8')
        # Build content block
        block: MessageContentBlock = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:{mime};base64,{b64}',
                'detail': 'auto'
            }
        }
        # Initialize base class with the block
        super().__init__(content=block)