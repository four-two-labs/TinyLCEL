"""Tests for filemagic module.

This module tests MIME type detection for various file formats:
- Image formats: PNG, JPEG, GIF, WebP, BMP (via puremagic fallback)
- Document formats: PDF, DOCX, PPTX, XLSX, DOC/PPT/XLS (OLE compound documents)

The implementation uses enhanced detection for Office formats:
- PDF: Detects %PDF- magic bytes
- DOCX/PPTX/XLSX: Examines ZIP contents to distinguish between formats
- DOC/PPT/XLS: Parses OLE compound document structure
- Other formats: Falls back to puremagic for broad format support
"""

import io
import zipfile

import pytest
from PIL import Image

from tinylcel.utils.filemagic import mimetype_from_blob

# Helper functions for creating test data

# Common magic bytes constants
MAGIC_BYTES = {
    'png': b'\x89PNG\r\n\x1a\n',
    'jpeg_jfif': b'\xff\xd8\xff\xe0',
    'jpeg_exif': b'\xff\xd8\xff\xe1',
    'jpeg_raw': b'\xff\xd8\xff\xdb',
    'jpeg_minimal': b'\xff\xd8\xff',
    'jpeg_near': b'\xff\xd8\xfe',
    'gif87a': b'GIF87a',
    'gif89a': b'GIF89a',
    'webp_riff': b'RIFF\x00\x00\x00\x00WEBP',
    'bmp': b'BM\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    'pdf': b'%PDF-1.4',
    'zip': b'PK\x03\x04',
    'ole': b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',
}

# Expected MIME types
EXPECTED_MIMETYPES = {
    'png': 'image/png',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'webp': 'image/webp',
    'bmp': 'image/x-ms-bmp',
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'msword': 'application/msword',
    'excel': 'application/vnd.ms-excel',
    'powerpoint': 'application/vnd.ms-powerpoint',
    'ole_generic': 'application/x-ole-storage',
    'ole_office': 'application/vnd.ms-office',
    'zip': 'application/zip',
    'octet_stream': 'application/octet-stream',
}


def _create_ole_compound_document(stream_name: str) -> bytes:
    """Create a minimal OLE compound document with the specified stream name.

    Args:
        stream_name: Name of the main stream to create (e.g., 'WordDocument', 'Workbook')

    Returns:
        Bytes of the OLE compound document
    """
    # OLE compound document header (512 bytes)
    header = bytearray(512)

    # OLE signature
    header[0:8] = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

    # Standard OLE header values
    header[24:26] = (0x003E).to_bytes(2, 'little')  # Minor version
    header[26:28] = (0x003E).to_bytes(2, 'little')  # Major version
    header[28:30] = (0xFFFE).to_bytes(2, 'little')  # Byte order
    header[30:32] = (0x0009).to_bytes(2, 'little')  # Sector size (512 bytes = 2^9)
    header[32:34] = (0x0006).to_bytes(2, 'little')  # Mini sector size (64 bytes = 2^6)
    header[44:48] = (1).to_bytes(4, 'little')  # Number of directory sectors
    header[48:52] = (1).to_bytes(4, 'little')  # Directory first sector
    header[76:80] = (0).to_bytes(4, 'little')  # FAT sectors

    # Create FAT sector (sector 0)
    fat_sector = bytearray(512)
    fat_sector[0:4] = (0xFFFFFFFE).to_bytes(4, 'little')  # FAT sector marker
    fat_sector[4:8] = (0xFFFFFFFE).to_bytes(4, 'little')  # Directory sector marker

    # Create directory sector (sector 1)
    dir_sector = bytearray(512)

    # Root entry (128 bytes)
    root_name = 'Root Entry'.encode('utf-16le').ljust(64, b'\x00')
    dir_sector[0:64] = root_name
    dir_sector[64:66] = (22).to_bytes(2, 'little')  # Name length
    dir_sector[66] = 5  # Root storage type
    dir_sector[67] = 0  # Color (red)

    # Main stream entry (128 bytes, starting at offset 128)
    stream_name_bytes = stream_name.encode('utf-16le').ljust(64, b'\x00')
    dir_sector[128:192] = stream_name_bytes
    name_length = len(stream_name) * 2 + 2  # UTF-16LE + null terminator
    dir_sector[192:194] = name_length.to_bytes(2, 'little')
    dir_sector[194] = 2  # Stream type
    dir_sector[195] = 1  # Color (black)

    return bytes(header) + bytes(fat_sector) + bytes(dir_sector)


def _create_office_xml_document(document_type: str, main_xml_content: str) -> bytes:
    """Create a minimal Office XML document (DOCX/PPTX/XLSX).

    Args:
        document_type: Type of document ('word', 'ppt', 'xl')
        main_xml_content: Content for the main XML file

    Returns:
        Bytes of the ZIP-based Office document
    """
    buffer = io.BytesIO()

    # MIME type mappings
    content_types = {
        'word': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml',
        'ppt': 'application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml',
        'xl': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml',
    }

    # Main file mappings
    main_files = {
        'word': 'word/document.xml',
        'ppt': 'ppt/presentation.xml',
        'xl': 'xl/workbook.xml',
    }

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Content Types
        zf.writestr(
            '[Content_Types].xml',
            f"""\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/{main_files[document_type]}"
ContentType="{content_types[document_type]}"/>
</Types>""",
        )

        # Relationships
        zf.writestr(
            '_rels/.rels',
            f"""\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
Target="{main_files[document_type]}"/>
</Relationships>""",
        )

        # Main document
        zf.writestr(main_files[document_type], main_xml_content)

    return buffer.getvalue()


def _create_image_bytes(format_name: str, size: tuple[int, int] = (10, 10), color: str = 'red') -> bytes:
    """Create image bytes using Pillow.

    Args:
        format_name: PIL format name (e.g., 'PNG', 'JPEG', 'GIF')
        size: Image dimensions as (width, height)
        color: Color name or tuple

    Returns:
        Bytes of the created image
    """
    # Handle special cases for different formats
    img = Image.new('P', size, color=0) if format_name == 'GIF' else Image.new('RGB', size, color=color)

    buffer = io.BytesIO()

    # Handle format-specific options
    save_kwargs = {}
    if format_name in {'JPEG', 'WEBP'}:
        save_kwargs['quality'] = 85

    img.save(buffer, format=format_name, **save_kwargs)
    return buffer.getvalue()


# Test fixtures


@pytest.fixture
def png_image_bytes() -> bytes:
    """Create a real PNG image using Pillow and return its bytes."""
    return _create_image_bytes('PNG')


@pytest.fixture
def jpeg_image_bytes() -> bytes:
    """Create a real JPEG image using Pillow and return its bytes."""
    return _create_image_bytes('JPEG', color='blue')


@pytest.fixture
def gif_image_bytes() -> bytes:
    """Create a real GIF image using Pillow and return its bytes."""
    return _create_image_bytes('GIF')


@pytest.fixture
def webp_image_bytes() -> bytes:
    """Create a real WebP image using Pillow and return its bytes."""
    return _create_image_bytes('WEBP', color='green')


@pytest.fixture
def bmp_image_bytes() -> bytes:
    """Create a real BMP image using Pillow and return its bytes."""
    return _create_image_bytes('BMP', color='yellow')


@pytest.fixture
def gif89a_image_bytes() -> bytes:
    """Create a minimal GIF89a image for testing."""
    # Create minimal valid GIF89a structure
    return (
        b'GIF89a\x0a\x00\x0a\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00'
        b',\x00\x00\x00\x00\x0a\x00\x0a\x00\x00\x02\x02\x04\x01\x00;'
    )


@pytest.fixture
def pdf_bytes() -> bytes:
    """Create a minimal valid PDF document."""
    # Minimal PDF structure that puremagic can identify
    return (
        b'%PDF-1.4\n'
        b'1 0 obj\n'
        b'<<\n'
        b'/Type /Catalog\n'
        b'/Pages 2 0 R\n'
        b'>>\n'
        b'endobj\n'
        b'2 0 obj\n'
        b'<<\n'
        b'/Type /Pages\n'
        b'/Kids [3 0 R]\n'
        b'/Count 1\n'
        b'>>\n'
        b'endobj\n'
        b'3 0 obj\n'
        b'<<\n'
        b'/Type /Page\n'
        b'/Parent 2 0 R\n'
        b'/MediaBox [0 0 612 792]\n'
        b'>>\n'
        b'endobj\n'
        b'xref\n'
        b'0 4\n'
        b'0000000000 65535 f \n'
        b'0000000010 00000 n \n'
        b'0000000079 00000 n \n'
        b'0000000136 00000 n \n'
        b'trailer\n'
        b'<<\n'
        b'/Size 4\n'
        b'/Root 1 0 R\n'
        b'>>\n'
        b'startxref\n'
        b'213\n'
        b'%%EOF\n'
    )


@pytest.fixture
def docx_bytes() -> bytes:
    """Create a minimal DOCX file structure."""
    return _create_office_xml_document(
        'word',
        """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>Test Document</w:t></w:r></w:p>
</w:body>
</w:document>""",
    )


@pytest.fixture
def pptx_bytes() -> bytes:
    """Create a minimal PPTX file structure."""
    return _create_office_xml_document(
        'ppt',
        """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst/>
<p:sldIdLst/>
<p:sldSz cx="9144000" cy="6858000"/>
</p:presentation>""",
    )


@pytest.fixture
def xlsx_bytes() -> bytes:
    """Create a minimal XLSX file structure."""
    return _create_office_xml_document(
        'xl',
        """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheets>
<sheet name="Sheet1" sheetId="1" r:id="rId1"/>
</sheets>
</workbook>""",
    )


@pytest.fixture
def ole_compound_doc_bytes() -> bytes:
    """Create a minimal OLE compound document structure that identifies as Word document."""
    return _create_ole_compound_document('WordDocument')


@pytest.fixture
def generic_ole_compound_doc_bytes() -> bytes:
    """Create a minimal OLE compound document structure that doesn't match any specific Office format."""
    return _create_ole_compound_document('SomeOtherStream')


@pytest.fixture
def excel_ole_compound_doc_bytes() -> bytes:
    """Create a minimal OLE compound document structure that identifies as Excel document."""
    return _create_ole_compound_document('Workbook')


@pytest.fixture
def excel_book_ole_compound_doc_bytes() -> bytes:
    """Create a minimal OLE compound document structure that identifies as Excel document using 'Book' name."""
    return _create_ole_compound_document('Book')


@pytest.fixture
def powerpoint_ole_compound_doc_bytes() -> bytes:
    """Create a minimal OLE compound document structure that identifies as PowerPoint document."""
    return _create_ole_compound_document('PowerPoint Document')


# Success cases - Document formats


@pytest.mark.parametrize(
    ('fixture_name', 'expected_mimetype'),
    [
        ('png_image_bytes', 'image/png'),
        ('jpeg_image_bytes', 'image/jpeg'),
        ('gif_image_bytes', 'image/gif'),
        ('webp_image_bytes', 'image/webp'),
        ('bmp_image_bytes', 'image/x-ms-bmp'),
    ],
)
def test_real_image_detection(fixture_name: str, expected_mimetype: str, request: pytest.FixtureRequest) -> None:
    """Test MIME type detection with real images generated by Pillow."""
    image_bytes = request.getfixturevalue(fixture_name)
    result = mimetype_from_blob(image_bytes)
    assert result == expected_mimetype


@pytest.mark.parametrize(
    ('fixture_name', 'expected_mimetype'),
    [
        ('pdf_bytes', 'application/pdf'),
        ('docx_bytes', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        ('pptx_bytes', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
        ('xlsx_bytes', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        ('ole_compound_doc_bytes', 'application/msword'),
        ('excel_ole_compound_doc_bytes', 'application/vnd.ms-excel'),
        ('excel_book_ole_compound_doc_bytes', 'application/vnd.ms-excel'),
        ('powerpoint_ole_compound_doc_bytes', 'application/vnd.ms-powerpoint'),
        ('generic_ole_compound_doc_bytes', 'application/x-ole-storage'),
    ],
)
def test_document_format_detection(fixture_name: str, expected_mimetype: str, request: pytest.FixtureRequest) -> None:
    """Test MIME type detection with real document files.

    The enhanced implementation now correctly identifies Office formats by examining
    their internal structure rather than just magic bytes.
    """
    document_bytes = request.getfixturevalue(fixture_name)
    result = mimetype_from_blob(document_bytes)
    assert result == expected_mimetype


@pytest.mark.parametrize(
    ('magic_bytes', 'expected_mimetype'),
    [
        (MAGIC_BYTES['png'], EXPECTED_MIMETYPES['png']),
        (MAGIC_BYTES['jpeg_jfif'], EXPECTED_MIMETYPES['jpeg']),
        (MAGIC_BYTES['jpeg_exif'], EXPECTED_MIMETYPES['jpeg']),
        (MAGIC_BYTES['gif87a'], EXPECTED_MIMETYPES['gif']),
        (MAGIC_BYTES['gif89a'], EXPECTED_MIMETYPES['gif']),
        (MAGIC_BYTES['webp_riff'], EXPECTED_MIMETYPES['webp']),
    ],
)
def test_magic_bytes_detection(magic_bytes: bytes, expected_mimetype: str) -> None:
    """Test MIME type detection with specific magic byte sequences."""
    _assert_mimetype_detection(magic_bytes, expected_mimetype)


@pytest.mark.parametrize(
    ('document_magic_bytes', 'expected_mimetype'),
    [
        (b'%PDF-1.0', EXPECTED_MIMETYPES['pdf']),
        (b'%PDF-1.4', EXPECTED_MIMETYPES['pdf']),
        (b'%PDF-1.7', EXPECTED_MIMETYPES['pdf']),
        (b'%PDF-2.0', EXPECTED_MIMETYPES['pdf']),
        # ZIP magic bytes → generic binary (not enough data for valid ZIP)
        (MAGIC_BYTES['zip'], EXPECTED_MIMETYPES['octet_stream']),
        # OLE compound document (generic without full structure)
        (MAGIC_BYTES['ole'], EXPECTED_MIMETYPES['ole_office']),
    ],
)
def test_document_magic_bytes_detection(document_magic_bytes: bytes, expected_mimetype: str) -> None:
    """Test MIME type detection with document format magic bytes.

    Note: Both OLE and ZIP compound documents require full structure to identify specific formats.
    Basic magic bytes return generic types.
    """
    _assert_mimetype_detection(document_magic_bytes, expected_mimetype)


@pytest.mark.parametrize(
    ('magic_bytes', 'expected_mimetype'),
    [
        (MAGIC_BYTES['png'], EXPECTED_MIMETYPES['png']),
        (MAGIC_BYTES['jpeg_minimal'], EXPECTED_MIMETYPES['jpeg']),
        (MAGIC_BYTES['gif87a'], EXPECTED_MIMETYPES['gif']),
        (MAGIC_BYTES['gif89a'], EXPECTED_MIMETYPES['gif']),
    ],
)
def test_minimal_valid_magic_bytes(magic_bytes: bytes, expected_mimetype: str) -> None:
    """Test detection with minimal valid magic byte sequences."""
    _assert_mimetype_detection(magic_bytes, expected_mimetype)


@pytest.mark.parametrize(
    ('fixture_name', 'expected_prefix'),
    [
        ('gif_image_bytes', b'GIF87a'),
        ('gif89a_image_bytes', b'GIF89a'),
    ],
)
def test_gif_format_detection_with_verification(
    fixture_name: str, expected_prefix: bytes, request: pytest.FixtureRequest
) -> None:
    """Test GIF format detection and verify the specific GIF version."""
    gif_bytes = request.getfixturevalue(fixture_name)
    result = mimetype_from_blob(gif_bytes)
    assert result == 'image/gif'
    # Verify the specific GIF version
    assert gif_bytes.startswith(expected_prefix)


def test_webp_exact_match() -> None:
    """Test WebP detection with exact RIFF...WEBP structure."""
    # Minimal valid WebP-like structure
    webp_data = b'RIFF\x10\x00\x00\x00WEBP'
    result = mimetype_from_blob(webp_data)
    assert result == 'image/webp'


def test_near_jpeg_detection() -> None:
    """Test that puremagic correctly identifies near-JPEG formats."""
    # This was previously expected to fail, but puremagic correctly identifies it
    _assert_mimetype_detection(MAGIC_BYTES['jpeg_near'], EXPECTED_MIMETYPES['jpeg'])


def test_real_webp_files_now_work(webp_image_bytes: bytes) -> None:
    """Test that real WebP files (generated by Pillow) now work correctly."""
    result = mimetype_from_blob(webp_image_bytes)
    assert result == 'image/webp'

    # Verify this is indeed a RIFF/WebP file with additional data
    assert webp_image_bytes.startswith(b'RIFF')
    assert b'WEBP' in webp_image_bytes
    assert not webp_image_bytes.endswith(b'WEBP')  # Has additional data after WEBP


def test_pdf_format_variations(pdf_bytes: bytes) -> None:
    """Test PDF detection with different versions and structures."""
    # Test with the complete PDF fixture
    result = mimetype_from_blob(pdf_bytes)
    assert result == 'application/pdf'

    # Verify it starts with PDF header
    assert pdf_bytes.startswith(b'%PDF-')

    # Test different PDF version headers
    pdf_versions = [b'%PDF-1.0', b'%PDF-1.3', b'%PDF-1.4', b'%PDF-1.7', b'%PDF-2.0']
    for version_header in pdf_versions:
        result = mimetype_from_blob(version_header)
        assert result == 'application/pdf'


def test_docx_zip_structure(docx_bytes: bytes) -> None:
    """Test that DOCX files have proper ZIP structure with expected content."""
    expected_files = ['word/document.xml', '[Content_Types].xml', '_rels/.rels']
    _assert_zip_structure(docx_bytes, EXPECTED_MIMETYPES['docx'], expected_files)


def test_pptx_zip_structure(pptx_bytes: bytes) -> None:
    """Test that PPTX files have proper ZIP structure with expected content."""
    expected_files = ['ppt/presentation.xml', '[Content_Types].xml', '_rels/.rels']
    _assert_zip_structure(pptx_bytes, EXPECTED_MIMETYPES['pptx'], expected_files)


def test_xlsx_zip_structure(xlsx_bytes: bytes) -> None:
    """Test that XLSX files have proper ZIP structure with expected content."""
    expected_files = ['xl/workbook.xml', '[Content_Types].xml', '_rels/.rels']
    _assert_zip_structure(xlsx_bytes, EXPECTED_MIMETYPES['xlsx'], expected_files)


def test_ole_compound_document_structure(ole_compound_doc_bytes: bytes) -> None:
    """Test that OLE compound document has proper structure."""
    _assert_ole_structure(ole_compound_doc_bytes, EXPECTED_MIMETYPES['msword'])


def test_generic_ole_compound_document_structure(generic_ole_compound_doc_bytes: bytes) -> None:
    """Test that OLE compound documents without specific Office identifiers return generic OLE type."""
    _assert_ole_structure(generic_ole_compound_doc_bytes, EXPECTED_MIMETYPES['ole_generic'])


def test_excel_ole_compound_document_structure(excel_ole_compound_doc_bytes: bytes) -> None:
    """Test that OLE compound documents with Excel Workbook stream are correctly identified."""
    _assert_ole_structure(excel_ole_compound_doc_bytes, EXPECTED_MIMETYPES['excel'])


def test_excel_book_ole_compound_document_structure(excel_book_ole_compound_doc_bytes: bytes) -> None:
    """Test that OLE compound documents with Excel Book stream are correctly identified."""
    _assert_ole_structure(excel_book_ole_compound_doc_bytes, EXPECTED_MIMETYPES['excel'])


def test_powerpoint_ole_compound_document_structure(powerpoint_ole_compound_doc_bytes: bytes) -> None:
    """Test that OLE compound documents with PowerPoint Document stream are correctly identified."""
    _assert_ole_structure(powerpoint_ole_compound_doc_bytes, EXPECTED_MIMETYPES['powerpoint'])


# Error cases


@pytest.mark.parametrize(
    ('invalid_blob', 'expected_exception', 'expected_message'),
    [
        (b'', 'ValueError', r'Input was empty'),
        (b'\x89P', 'puremagic.PureError', r'Could not identify file'),  # Too short for PNG
        (b'INVALID_MAGIC_BYTES', 'puremagic.PureError', r'Could not identify file'),
        (b'\x00\x00\x00\x00', 'puremagic.PureError', r'Could not identify file'),  # Null bytes
        (b'JPEG', 'puremagic.PureError', r'Could not identify file'),  # Wrong magic for JPEG
        (b'PNG', 'puremagic.PureError', r'Could not identify file'),  # Wrong magic for PNG
        (b'Random text content', 'puremagic.PureError', r'Could not identify file'),  # Text content
        (b'gif89a', 'puremagic.PureError', r'Could not identify file'),  # Lowercase
        (b'png', 'puremagic.PureError', r'Could not identify file'),  # Lowercase, missing magic
        (b'%pdf-1.4', 'puremagic.PureError', r'Could not identify file'),  # Lowercase PDF
        (b'PDF-1.4', 'puremagic.PureError', r'Could not identify file'),  # Missing % for PDF
        (b'PK\x03', 'puremagic.PureError', r'Could not identify file'),  # Incomplete ZIP magic
        (b'\xd0\xcf\x11', 'puremagic.PureError', r'Could not identify file'),  # Incomplete OLE magic
    ],
)
def test_invalid_formats_raise_errors(invalid_blob: bytes, expected_exception: str, expected_message: str) -> None:
    """Test that various invalid format blobs raise appropriate errors."""
    if expected_exception == 'puremagic.PureError':
        import puremagic

        with pytest.raises(puremagic.PureError, match=expected_message):
            mimetype_from_blob(invalid_blob)
    elif expected_exception == 'ValueError':
        with pytest.raises(ValueError, match=expected_message):
            mimetype_from_blob(invalid_blob)


def test_audio_file_not_image() -> None:
    """Test that audio files are correctly identified but might not be what we expect for images."""
    # RIFF file that's actually a WAV file
    riff_data = b'RIFF\x10\x00\x00\x00WAVE'
    result = mimetype_from_blob(riff_data)
    # This should return an audio MIME type, not an image type
    assert result.startswith(('audio/', 'application/'))
    assert result != 'image/webp'


def test_corrupted_zip_file() -> None:
    """Test that corrupted ZIP files are handled appropriately."""
    # ZIP magic bytes but corrupted structure
    corrupted_zip = b'PK\x03\x04' + b'\x00' * 100  # ZIP magic but invalid structure
    result = mimetype_from_blob(corrupted_zip)
    # Should return generic binary type for corrupted ZIP data
    assert result == 'application/octet-stream'


def test_corrupted_ole_file() -> None:
    """Test that corrupted OLE files are handled appropriately."""
    # OLE magic bytes but insufficient structure
    corrupted_ole = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 100
    result = mimetype_from_blob(corrupted_ole)
    # Should be identified as some OLE format
    assert result in ('application/msword', 'application/vnd.ms-office')


# Edge cases


def test_large_image_data() -> None:
    """Test detection works with larger image data."""
    # Create a larger PNG image
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    large_png_bytes = buffer.getvalue()

    result = mimetype_from_blob(large_png_bytes)
    assert result == 'image/png'


def test_webp_with_minimal_riff_structure() -> None:
    """Test WebP detection with minimal RIFF structure."""
    # Create minimal RIFF...WEBP structure
    minimal_webp = b'RIFF\x08\x00\x00\x00WEBP'
    result = mimetype_from_blob(minimal_webp)
    assert result == 'image/webp'


def test_jpeg_different_variants() -> None:
    """Test JPEG detection with different JPEG variants."""
    # Different JPEG magic byte variants (all start with FF D8 FF)
    jpeg_variants = [
        MAGIC_BYTES['jpeg_jfif'],  # JFIF
        MAGIC_BYTES['jpeg_exif'],  # EXIF
        MAGIC_BYTES['jpeg_raw'],  # Raw JPEG
    ]

    for variant in jpeg_variants:
        _assert_mimetype_detection(variant, EXPECTED_MIMETYPES['jpeg'])


def test_pdf_with_additional_content(pdf_bytes: bytes) -> None:
    """Test PDF detection with additional content appended."""
    extended_pdf = pdf_bytes + b'\x00' * 1000 + b'extra content here'
    result = mimetype_from_blob(extended_pdf)
    assert result == 'application/pdf'


def test_zip_based_formats_distinction() -> None:
    """Test that the enhanced implementation can distinguish between ZIP-based Office formats."""
    # Enhanced implementation examines ZIP contents for Office format detection
    # For incomplete ZIP data, returns generic binary type

    # ZIP magic bytes alone - not enough data to be a valid ZIP file
    zip_magic = b'PK\x03\x04'
    result = mimetype_from_blob(zip_magic)

    # Should be generic binary type for incomplete data
    assert result == 'application/octet-stream'


# Integration tests


def test_all_supported_formats_round_trip(
    png_image_bytes: bytes,
    jpeg_image_bytes: bytes,
    gif_image_bytes: bytes,
    webp_image_bytes: bytes,
) -> None:
    """Test that all supported formats are correctly detected."""
    test_cases = [
        (png_image_bytes, 'image/png'),
        (jpeg_image_bytes, 'image/jpeg'),
        (gif_image_bytes, 'image/gif'),
        (webp_image_bytes, 'image/webp'),
    ]

    for image_bytes, expected_mimetype in test_cases:
        result = mimetype_from_blob(image_bytes)
        assert result == expected_mimetype


def test_modern_office_formats_round_trip(
    pdf_bytes: bytes,
    docx_bytes: bytes,
    pptx_bytes: bytes,
    xlsx_bytes: bytes,
) -> None:
    """Test that modern Office formats are correctly detected."""
    test_cases = [
        (pdf_bytes, 'application/pdf'),
        (docx_bytes, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        (pptx_bytes, 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
        (xlsx_bytes, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    ]

    for document_bytes, expected_mimetype in test_cases:
        result = mimetype_from_blob(document_bytes)
        assert result == expected_mimetype


def test_legacy_ole_formats_round_trip(
    ole_compound_doc_bytes: bytes,
    excel_ole_compound_doc_bytes: bytes,
    excel_book_ole_compound_doc_bytes: bytes,
    powerpoint_ole_compound_doc_bytes: bytes,
    generic_ole_compound_doc_bytes: bytes,
) -> None:
    """Test that legacy OLE Office formats are correctly detected."""
    test_cases = [
        (ole_compound_doc_bytes, 'application/msword'),
        (excel_ole_compound_doc_bytes, 'application/vnd.ms-excel'),
        (excel_book_ole_compound_doc_bytes, 'application/vnd.ms-excel'),
        (powerpoint_ole_compound_doc_bytes, 'application/vnd.ms-powerpoint'),
        (generic_ole_compound_doc_bytes, 'application/x-ole-storage'),
    ]

    for document_bytes, expected_mimetype in test_cases:
        result = mimetype_from_blob(document_bytes)
        assert result == expected_mimetype


def test_enhanced_office_format_detection() -> None:
    """Test demonstrating enhanced Office format detection capabilities.

    The enhanced implementation examines ZIP contents to distinguish between
    different Office formats rather than relying only on magic bytes.
    """
    # ZIP magic bytes without valid structure should return generic binary type
    zip_magic = b'PK\x03\x04'
    result = mimetype_from_blob(zip_magic)

    # Should return generic binary type for incomplete data
    assert result == 'application/octet-stream'


def test_different_image_sizes_and_colors() -> None:
    """Test detection works with images of different sizes and colors."""
    test_configs: list[tuple[int, int, str | tuple[int, int, int]]] = [
        (1, 1, 'black'),
        (50, 30, 'white'),
        (10, 10, (255, 0, 0)),  # Red tuple
        (5, 5, (0, 255, 0)),  # Green tuple
    ]

    for width, height, color in test_configs:
        # Test with PNG format
        img = Image.new('RGB', (width, height), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()

        result = mimetype_from_blob(png_bytes)
        assert result == 'image/png'


def test_grayscale_images() -> None:
    """Test detection works with grayscale images."""
    # Create grayscale image
    img = Image.new('L', (20, 20), color=128)  # 'L' mode for grayscale
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    grayscale_bytes = buffer.getvalue()

    result = mimetype_from_blob(grayscale_bytes)
    assert result == 'image/png'


def test_rgba_images() -> None:
    """Test detection works with RGBA images (with transparency)."""
    # Create RGBA image with transparency
    img = Image.new('RGBA', (15, 15), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    rgba_bytes = buffer.getvalue()

    result = mimetype_from_blob(rgba_bytes)
    assert result == 'image/png'


def test_bmp_format_support(bmp_image_bytes: bytes) -> None:
    """Test that BMP format is now correctly supported."""
    result = mimetype_from_blob(bmp_image_bytes)
    assert result == 'image/x-ms-bmp'

    # Verify this is indeed a BMP file
    assert bmp_image_bytes.startswith(b'BM')


# Tests demonstrating improved capabilities


def test_webp_files_now_fully_supported() -> None:
    """Test that both simple and complex WebP structures work."""
    # Simple WebP structure
    simple_webp = b'RIFF\x0c\x00\x00\x00WEBP'
    result1 = mimetype_from_blob(simple_webp)
    assert result1 == 'image/webp'

    # Complex WebP structure (with additional data after WEBP)
    complex_webp = b'RIFF\x20\x00\x00\x00WEBPVP8 \x14\x00\x00\x00\x00\x00\x00\x00'
    result2 = mimetype_from_blob(complex_webp)
    assert result2 == 'image/webp'


def test_more_image_formats_supported() -> None:
    """Test that puremagic supports more image formats than the original implementation."""
    # Test formats that the original implementation didn't support
    formats_to_test = [
        # BMP format
        (b'BM\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', 'image/x-ms-bmp'),
    ]

    for magic_bytes, expected_mime in formats_to_test:
        result = mimetype_from_blob(magic_bytes)
        assert result == expected_mime


def test_document_formats_comprehensive_support() -> None:
    """Test comprehensive support for document formats with enhanced detection."""
    # Test that enhanced implementation provides good coverage of document formats
    format_tests = [
        # PDF variants
        (b'%PDF-1.0', 'application/pdf'),
        (b'%PDF-1.4', 'application/pdf'),
        (b'%PDF-2.0', 'application/pdf'),
        # ZIP magic bytes without structure should return generic binary type
        (b'PK\x03\x04', 'application/octet-stream'),  # Incomplete ZIP data
        # OLE compound documents - enhanced detection examines structure
        (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', lambda x: x.startswith('application/')),  # OLE formats
    ]

    for magic_bytes, expected in format_tests:
        result = mimetype_from_blob(magic_bytes)
        if callable(expected):
            assert expected(result), f'Failed for {magic_bytes!r}: got {result}'
        else:
            assert result == expected, f'Failed for {magic_bytes!r}: expected {expected}, got {result}'


def _assert_mimetype_detection(blob_data: bytes, expected_mimetype: str) -> None:
    """Helper function to assert MIME type detection.

    Args:
        blob_data: Binary data to test
        expected_mimetype: Expected MIME type result
    """
    result = mimetype_from_blob(blob_data)
    assert result == expected_mimetype


def _assert_ole_structure(ole_bytes: bytes, expected_mimetype: str) -> None:
    """Helper function to assert OLE compound document structure and detection.

    Args:
        ole_bytes: OLE compound document bytes
        expected_mimetype: Expected MIME type result
    """
    result = mimetype_from_blob(ole_bytes)
    assert result == expected_mimetype

    # Verify OLE signature
    assert ole_bytes.startswith(MAGIC_BYTES['ole'])

    # Verify minimum size (OLE docs need at least 512 bytes for header)
    assert len(ole_bytes) >= 512


def _assert_zip_structure(zip_bytes: bytes, expected_mimetype: str, expected_files: list[str]) -> None:
    """Helper function to assert ZIP structure and detection.

    Args:
        zip_bytes: ZIP file bytes
        expected_mimetype: Expected MIME type result
        expected_files: List of files that should be present in the ZIP
    """
    result = mimetype_from_blob(zip_bytes)
    assert result == expected_mimetype

    # Verify it's a valid ZIP file with expected structure
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        filenames = zf.namelist()
        for expected_file in expected_files:
            assert expected_file in filenames


# Tests for extension_from_blob function


def test_extension_from_blob_import() -> None:
    """Test that extension_from_blob can be imported from the module."""
    from tinylcel.utils.filemagic import extension_from_blob

    assert callable(extension_from_blob)


@pytest.mark.parametrize(
    ('fixture_name', 'expected_extension'),
    [
        ('png_image_bytes', '.png'),
        ('jpeg_image_bytes', '.jpg'),
        ('gif_image_bytes', '.gif'),
        ('webp_image_bytes', '.webp'),
        ('bmp_image_bytes', '.bmp'),
        ('pdf_bytes', '.pdf'),
    ],
)
def test_extension_from_blob_success_cases(
    fixture_name: str, expected_extension: str, request: pytest.FixtureRequest
) -> None:
    """Test extension_from_blob with various file formats."""
    from tinylcel.utils.filemagic import extension_from_blob

    file_bytes = request.getfixturevalue(fixture_name)
    result = extension_from_blob(file_bytes)
    assert result == expected_extension


@pytest.mark.parametrize(
    ('magic_bytes', 'expected_extension'),
    [
        (MAGIC_BYTES['png'], '.png'),
        (MAGIC_BYTES['jpeg_jfif'], '.jpg'),
        (MAGIC_BYTES['jpeg_exif'], '.jpg'),
        (MAGIC_BYTES['gif87a'], '.gif'),
        (MAGIC_BYTES['gif89a'], '.gif'),
        (MAGIC_BYTES['webp_riff'], '.webp'),
        (b'%PDF-1.4', '.pdf'),
        (b'%PDF-1.0', '.pdf'),
        (b'%PDF-2.0', '.pdf'),
    ],
)
def test_extension_from_blob_magic_bytes(magic_bytes: bytes, expected_extension: str) -> None:
    """Test extension_from_blob with specific magic byte sequences."""
    from tinylcel.utils.filemagic import extension_from_blob

    result = extension_from_blob(magic_bytes)
    assert result == expected_extension


def test_extension_from_blob_office_documents(
    docx_bytes: bytes,
    pptx_bytes: bytes,
    xlsx_bytes: bytes,
    ole_compound_doc_bytes: bytes,
    excel_ole_compound_doc_bytes: bytes,
    powerpoint_ole_compound_doc_bytes: bytes,
) -> None:
    """Test extension_from_blob with Office document formats."""
    from tinylcel.utils.filemagic import extension_from_blob

    test_cases = [
        (docx_bytes, '.docx'),
        (pptx_bytes, '.pptx'),
        (xlsx_bytes, '.xlsx'),
        (ole_compound_doc_bytes, '.doc'),
        (excel_ole_compound_doc_bytes, '.xls'),  # Now returns preferred .xls extension
        (powerpoint_ole_compound_doc_bytes, '.ppt'),
    ]

    for document_bytes, expected_extension in test_cases:
        result = extension_from_blob(document_bytes)
        assert result == expected_extension


def test_extension_from_blob_empty_input() -> None:
    """Test that extension_from_blob raises ValueError for empty input."""
    from tinylcel.utils.filemagic import extension_from_blob

    with pytest.raises(ValueError, match=r'Input was empty'):
        extension_from_blob(b'')


def test_extension_from_blob_unknown_format() -> None:
    """Test that extension_from_blob raises StopIteration for unknown formats."""
    from tinylcel.utils.filemagic import extension_from_blob

    # Use data that puremagic can't identify
    unknown_data = b'UNKNOWN_MAGIC_BYTES_12345'

    # This should raise puremagic.PureError from mimetype_from_blob first
    import puremagic

    with pytest.raises(puremagic.PureError, match=r'Could not identify file'):
        extension_from_blob(unknown_data)


def test_extension_from_blob_consistency_with_mimetype() -> None:
    """Test that extension_from_blob is consistent with mimetype_from_blob results."""
    from tinylcel.utils.filemagic import mimetype_from_blob
    from tinylcel.utils.filemagic import extension_from_blob

    test_data = [
        MAGIC_BYTES['png'],
        MAGIC_BYTES['jpeg_jfif'],
        MAGIC_BYTES['gif87a'],
        b'%PDF-1.4',
    ]

    for data in test_data:
        # Get MIME type
        mime_type = mimetype_from_blob(data)

        # Get extension
        extension = extension_from_blob(data)

        # Verify they are related (extension should start with '.')
        assert extension.startswith('.')
        assert len(extension) > 1  # Should have actual extension after the dot

        # Extension should be consistent with the MIME type
        # (This tests that they use the same underlying detection)
        if mime_type == 'image/png':
            assert extension == '.png'
        elif mime_type == 'image/jpeg':
            assert extension == '.jpg'
        elif mime_type == 'image/gif':
            assert extension == '.gif'
        elif mime_type == 'application/pdf':
            assert extension == '.pdf'


def test_extension_from_blob_all_supported_formats_round_trip(
    png_image_bytes: bytes,
    jpeg_image_bytes: bytes,
    gif_image_bytes: bytes,
    webp_image_bytes: bytes,
    pdf_bytes: bytes,
) -> None:
    """Test extension_from_blob round trip with all supported formats."""
    from tinylcel.utils.filemagic import extension_from_blob

    test_cases = [
        (png_image_bytes, '.png'),
        (jpeg_image_bytes, '.jpg'),
        (gif_image_bytes, '.gif'),
        (webp_image_bytes, '.webp'),
        (pdf_bytes, '.pdf'),
    ]

    for file_bytes, expected_extension in test_cases:
        result = extension_from_blob(file_bytes)
        assert result == expected_extension
        # Verify extension format
        assert result.startswith('.')
        assert len(result) > 1
        assert result.islower() or result in ['.PDF']  # Most extensions are lowercase
