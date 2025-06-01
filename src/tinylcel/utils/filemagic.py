import io
import zipfile
import typing as t
from itertools import chain

import puremagic  # type: ignore[import-untyped]

# Preferred extensions for common MIME types
# Maps MIME types to their most commonly expected file extensions
g_preferred_extensions: t.Final[dict[str, str]] = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/x-ms-bmp': '.bmp',
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/msword': '.doc',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.ms-powerpoint': '.ppt',
    'application/zip': '.zip',
    'text/plain': '.txt',
    'text/html': '.html',
    'text/css': '.css',
    'application/javascript': '.js',
    'application/json': '.json',
    'application/xml': '.xml',
}


def _parse_ole_compound_document(blob: bytes) -> str:
    """Parse OLE compound document to identify specific Office format.

    Returns specific MIME type for Word, Excel, PowerPoint documents,
    or generic OLE type if the specific format cannot be determined.
    """
    ole_names: t.Final[list[tuple[tuple[bytes, ...], str]]] = [
        (('WordDocument'.encode('utf-16le'),), 'application/msword'),
        (('Workbook'.encode('utf-16le'), 'Book'.encode('utf-16le')), 'application/vnd.ms-excel'),
        (('PowerPoint Document'.encode('utf-16le'),), 'application/vnd.ms-powerpoint'),
    ]

    if len(blob) < 512:
        return 'application/vnd.ms-office'

    # Parse header
    header: t.Final[bytes] = blob[:512]
    sector_shift: t.Final[int] = int.from_bytes(header[30:32], 'little')
    sector_size: t.Final[int] = 1 << sector_shift
    dir_first_sector: t.Final[int] = int.from_bytes(header[48:52], 'little')

    dir_offset: t.Final[int] = (dir_first_sector + 1) * sector_size
    if dir_offset >= len(blob):
        return 'application/vnd.ms-office'

    # Read directory entries (each entry is 128 bytes)
    pos = dir_offset
    while pos + 128 <= len(blob):
        entry = blob[pos : pos + 128]
        name_bytes = entry[:64]

        for names, mime_type in ole_names:
            if name_bytes.startswith(names):
                return mime_type

        pos += 128

    # If we can't identify it specifically, return generic OLE
    return 'application/x-ole-storage'


def _parse_zip_file(blob: bytes) -> str:
    """Parse ZIP file to identify specific Office format.

    Returns specific MIME type for Word, Excel, PowerPoint documents,
    or a generic ZIP type if the specific format cannot be determined.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zip_file:
            files = set(zip_file.namelist())
            if 'word/document.xml' in files:
                return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            if 'ppt/presentation.xml' in files:
                return 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            if 'xl/workbook.xml' in files:
                return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    except zipfile.BadZipFile:
        # If we can't parse as a valid ZIP file, it's not any ZIP-based format
        # Return generic binary type for incomplete/corrupted data
        return 'application/octet-stream'

    return 'application/zip'


def mimetype_from_blob(blob: bytes) -> str:
    """Determine MIME type from binary data.

    Uses enhanced detection for Office formats (PDF, DOCX/PPTX/XLSX, DOC/PPT/XLS)
    and falls back to puremagic for other formats.
    """
    if not blob:
        raise ValueError('Input was empty')

    mime_type: str = puremagic.from_string(blob, mime=True)

    match mime_type:
        case (
            'application/zip'
            | 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            | 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            | 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ):
            return _parse_zip_file(blob)
        case 'application/x-ole-storage' | 'application/msword':
            return _parse_ole_compound_document(blob)
        case _:
            return mime_type

    return mime_type


def extension_from_blob(blob: bytes) -> str:
    """Get file extension from binary data based on detected MIME type.

    Args:
        blob: Binary data to analyze

    Returns:
        File extension string (e.g., '.pdf', '.docx')

    Raises:
        ValueError: If input is empty
        StopIteration: If no extension found for the detected MIME type
    """
    mime_type: t.Final[str] = mimetype_from_blob(blob)

    if mime_type in g_preferred_extensions:
        return g_preferred_extensions[mime_type]

    return next(
        x.extension
        for x in chain(puremagic.magic_header_array, puremagic.magic_footer_array)
        if x.mime_type == mime_type and x.extension
    )
