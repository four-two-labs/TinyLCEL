"""ImageChunk: a HumanMessageChunk subclass that wraps an image as a base64 data-URL block."""

import io
import base64
from pathlib import Path

try:
    import magic  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "The 'python-magic' and 'pillow' libraries are required to use ImageChunk. "
        "Please install them with 'pip install python-magic pillow'."
    ) from e

from tinylcel.messages.chunks import HumanMessageChunk


def image_chunk(
    image: Path | str | Image.Image,
    wire_format: str = 'webp',
    quality: int = 85,
) -> HumanMessageChunk:
    """Create an image chunk for a human message."""

    def save_image(image: Image.Image) -> bytes:
        with io.BytesIO() as buffer:
            image.save(buffer, format=wire_format, quality=quality)
            buffer.seek(0)
            return buffer.read()

    image_data = save_image(image if isinstance(image, Image.Image) else Image.open(image))

    mime_type = magic.from_buffer(image_data, mime=True)
    base64_data = base64.b64encode(image_data).decode('u8')

    return HumanMessageChunk(
        content=[
            {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{base64_data}', 'detail': 'auto'}}
        ]
    )


ImageChunk = image_chunk
