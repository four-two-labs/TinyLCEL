"""ImageChunk: a HumanMessageChunk subclass that wraps an image as a base64 data-URL block."""

import io
import base64
from pathlib import Path

from PIL import Image  # type: ignore[import-not-found]

from tinylcel.messages.chunks import HumanMessageChunk
from tinylcel.utils.filemagic import mimetype_from_blob


def image_chunk(
    image: Path | str | Image.Image,
    wire_format: str = 'webp',
    quality: int = 85,
) -> HumanMessageChunk:
    """Create an image chunk for a human message.

    Converts an image to a base64-encoded data URL and wraps it in a HumanMessageChunk
    suitable for use in chat messages. The image is converted to the specified wire format
    and quality before encoding.

    Args:
        image: The input image. Can be a file path (Path or str) to an image file,
            or a PIL Image object.
        wire_format: The image format to use for encoding. Defaults to 'webp'.
            Common formats include 'webp', 'jpeg', 'png'.
        quality: The compression quality for lossy formats (1-100). Higher values
            mean better quality but larger file sizes. Defaults to 85. Only applies
            to formats that support quality settings (like JPEG, WebP).

    Returns:
        HumanMessageChunk: A message chunk containing the image as a base64 data URL
        with the appropriate MIME type.

    Raises:
        IOError: If the image file cannot be opened or read.
        ValueError: If an unsupported wire_format is specified.

    Examples:
        >>> from pathlib import Path
        >>> chunk = image_chunk('photo.jpg')
        >>> chunk = image_chunk(Path('diagram.png'), wire_format='png')
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), 'red')
        >>> chunk = image_chunk(img, wire_format='jpeg', quality=95)
    """

    def save_image(image: Image.Image) -> bytes:
        with io.BytesIO() as buffer:
            image.save(buffer, format=wire_format, quality=quality)
            buffer.seek(0)
            return buffer.read()

    image_data = save_image(image if isinstance(image, Image.Image) else Image.open(image))

    mime_type = mimetype_from_blob(image_data)
    base64_data = base64.b64encode(image_data).decode('u8')

    return HumanMessageChunk(
        content=[
            {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{base64_data}', 'detail': 'auto'}}
        ]
    )


ImageChunk = image_chunk
