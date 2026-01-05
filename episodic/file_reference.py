r"""
File reference processing for Episodic.

Handles @file syntax to include file contents in prompts.
Supports text files, PDFs (text extraction), and images (multimodal).

Syntax:
    @file.txt           - Read text file, inject contents
    @"path/with spaces" - Quoted path for spaces
    @file.pdf           - Extract text from PDF
    @file.png           - Send as base64 image (multimodal)
    @file.pdf:vision    - Render PDF pages as images
    @file.pdf:vision:1-5 - Specific page range
    \@                  - Escape, literal @ character
"""

import base64
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from episodic.config import config

logger = logging.getLogger(__name__)

# File extensions considered as text
TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.rst', '.json', '.yaml', '.yml',
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.xml', '.toml', '.ini', '.cfg', '.conf', '.env',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
    '.sql', '.graphql', '.proto',
    '.csv', '.tsv', '.log',
}

# Image extensions for multimodal
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}

# PDF extension
PDF_EXTENSION = '.pdf'


@dataclass
class FileRef:
    """Represents a file reference found in text."""
    match_text: str      # Full match including @
    path: str            # The file path
    modifier: Optional[str] = None    # e.g., "vision"
    page_range: Optional[Tuple[int, int]] = None  # e.g., (1, 5) for :vision:1-5


def find_file_references(text: str) -> Tuple[List[FileRef], str]:
    r"""
    Find all @file references in text.

    Returns:
        Tuple of (list of FileRef objects, text with \@ converted to @)
    """
    refs = []

    # First, handle escaped \@ by temporarily replacing them
    escaped_placeholder = "\x00ESCAPED_AT\x00"
    text_processed = text.replace("\\@", escaped_placeholder)

    # Pattern: @"quoted path" OR @unquoted_path
    # Unquoted path can include :modifier:range
    # Exclude trailing punctuation like ? ! , . ; from unquoted paths
    pattern = r'(?<!\w)@(?:"([^"]+)"|([^\s?!,;]+))'

    for match in re.finditer(pattern, text_processed):
        match_text = match.group(0)
        # Group 1 is quoted path, group 2 is unquoted
        raw_path = match.group(1) if match.group(1) else match.group(2)

        # Parse modifiers from unquoted paths (e.g., file.pdf:vision:1-5)
        modifier = None
        page_range = None
        path = raw_path

        if match.group(2):  # Only parse modifiers for unquoted paths
            parts = raw_path.split(':')
            path = parts[0]

            if len(parts) >= 2 and parts[1] == 'vision':
                modifier = 'vision'
                if len(parts) >= 3:
                    # Parse page range like "1-5" or "3"
                    range_str = parts[2]
                    if '-' in range_str:
                        try:
                            start, end = range_str.split('-')
                            page_range = (int(start), int(end))
                        except ValueError:
                            pass  # Invalid range, ignore
                    else:
                        try:
                            page_num = int(range_str)
                            page_range = (page_num, page_num)
                        except ValueError:
                            pass

        refs.append(FileRef(
            match_text=match_text,
            path=path,
            modifier=modifier,
            page_range=page_range
        ))

    # Restore escaped @ symbols
    text_processed = text_processed.replace(escaped_placeholder, "@")

    return refs, text_processed


def resolve_path(path_str: str) -> Optional[Path]:
    """
    Resolve a file path (absolute or relative to CWD).

    Returns:
        Resolved Path if file exists, None otherwise
    """
    path = Path(path_str).expanduser()

    # If absolute, check directly
    if path.is_absolute():
        return path if path.exists() else None

    # Relative to CWD
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    return None


def get_file_type(path: Path) -> str:
    """
    Determine the file type category.

    Returns:
        'text', 'image', 'pdf', or 'unknown'
    """
    suffix = path.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        return 'text'
    elif suffix in IMAGE_EXTENSIONS:
        return 'image'
    elif suffix == PDF_EXTENSION:
        return 'pdf'
    else:
        # Try to guess from MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            if mime_type.startswith('text/'):
                return 'text'
            elif mime_type.startswith('image/'):
                return 'image'
            elif mime_type == 'application/pdf':
                return 'pdf'
        return 'unknown'


def process_text_file(path: Path) -> str:
    """Read a text file and return its contents."""
    max_size = config.get("file_ref_max_text_size", 100000)

    content = path.read_text(encoding='utf-8')

    if len(content) > max_size:
        content = content[:max_size]
        content += f"\n\n[... truncated at {max_size} characters ...]"

    return content


def process_pdf_text(path: Path) -> str:
    """
    Extract text from a PDF file.

    Uses pdfplumber by default, or Marker if configured.
    """
    extractor = config.get("pdf_extractor", "pdfplumber")

    # Try Marker if configured
    if extractor == "marker":
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(str(path))
            return rendered.markdown
        except ImportError:
            logger.warning("Marker not installed, falling back to pdfplumber")
        except Exception as e:
            logger.warning(f"Marker extraction failed: {e}, falling back to pdfplumber")

    # Default: pdfplumber
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        content = "\n\n".join(text_parts)

        # Apply size limit
        max_size = config.get("file_ref_max_text_size", 100000)
        if len(content) > max_size:
            content = content[:max_size]
            content += f"\n\n[... truncated at {max_size} characters ...]"

        return content

    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")


def process_image(path: Path) -> Dict[str, Any]:
    """
    Process an image file for multimodal LLM input.

    Returns:
        LiteLLM-compatible image content block
    """
    # Determine MIME type
    suffix = path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    mime_type = mime_types.get(suffix, 'image/png')

    # Read and encode
    image_data = path.read_bytes()
    base64_data = base64.b64encode(image_data).decode('utf-8')

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_data}"
        }
    }


def process_pdf_vision(path: Path, page_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """
    Convert PDF pages to images for multimodal LLM input.

    Args:
        path: Path to the PDF file
        page_range: Optional (start, end) page numbers (1-indexed)

    Returns:
        List of LiteLLM-compatible image content blocks
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError(
            "pdf2image not installed. Run: pip install pdf2image\n"
            "Also requires poppler:\n"
            "  macOS: brew install poppler\n"
            "  Ubuntu: apt-get install poppler-utils"
        )

    # Determine page range
    default_pages = config.get("file_ref_vision_pages", 5)

    if page_range:
        first_page, last_page = page_range
    else:
        first_page = 1
        last_page = default_pages

    try:
        # Convert PDF pages to images
        images = convert_from_path(
            path,
            first_page=first_page,
            last_page=last_page,
            dpi=150  # Balance quality vs size
        )
    except Exception as e:
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            raise RuntimeError(
                "pdf2image requires poppler. Install:\n"
                "  macOS: brew install poppler\n"
                "  Ubuntu: apt-get install poppler-utils"
            )
        raise

    # Convert each image to base64
    import io

    content_blocks = []
    for i, image in enumerate(images):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_data}"
            }
        })

    return content_blocks


def process_file_references(user_input: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Process all @file references in user input.

    This is the main entry point for file reference processing.

    Args:
        user_input: The user's input text containing @file references

    Returns:
        Tuple of:
        - Modified text with file contents injected (for text files)
        - List of multimodal content blocks (for images)
        - List of error messages
    """
    refs, text = find_file_references(user_input)

    if not refs:
        return user_input, [], []

    multimodal_blocks = []
    errors = []

    # Process each reference
    for ref in refs:
        try:
            resolved = resolve_path(ref.path)

            if resolved is None:
                errors.append(f"@{ref.path}: File not found")
                continue

            # Check permissions
            if not resolved.is_file():
                errors.append(f"@{ref.path}: Not a file")
                continue

            file_type = get_file_type(resolved)

            # Handle based on file type and modifiers
            if file_type == 'pdf' and ref.modifier == 'vision':
                # PDF as images
                try:
                    blocks = process_pdf_vision(resolved, ref.page_range)
                    multimodal_blocks.extend(blocks)
                    # Remove the reference from text
                    text = text.replace(ref.match_text, f"[PDF: {resolved.name}]")
                except RuntimeError as e:
                    errors.append(f"@{ref.path}: {str(e)}")

            elif file_type == 'pdf':
                # PDF as text
                try:
                    content = process_pdf_text(resolved)
                    replacement = f"\n\n--- Content of {resolved.name} ---\n{content}\n--- End of {resolved.name} ---\n\n"
                    text = text.replace(ref.match_text, replacement)
                except Exception as e:
                    errors.append(f"@{ref.path}: Failed to extract text ({str(e)}). Try :vision mode.")

            elif file_type == 'image':
                # Image as multimodal
                try:
                    block = process_image(resolved)
                    multimodal_blocks.append(block)
                    # Remove the reference from text
                    text = text.replace(ref.match_text, f"[Image: {resolved.name}]")
                except Exception as e:
                    errors.append(f"@{ref.path}: Failed to process image ({str(e)})")

            elif file_type == 'text':
                # Text file - inline
                try:
                    content = process_text_file(resolved)
                    replacement = f"\n\n--- Content of {resolved.name} ---\n{content}\n--- End of {resolved.name} ---\n\n"
                    text = text.replace(ref.match_text, replacement)
                except PermissionError:
                    errors.append(f"@{ref.path}: Permission denied")
                except Exception as e:
                    errors.append(f"@{ref.path}: Failed to read ({str(e)})")

            else:
                # Unknown type - try to read as text
                try:
                    content = process_text_file(resolved)
                    replacement = f"\n\n--- Content of {resolved.name} ---\n{content}\n--- End of {resolved.name} ---\n\n"
                    text = text.replace(ref.match_text, replacement)
                except Exception:
                    errors.append(f"@{ref.path}: Unknown file type, cannot process")

        except PermissionError:
            errors.append(f"@{ref.path}: Permission denied")
        except Exception as e:
            errors.append(f"@{ref.path}: Error ({str(e)})")

    return text, multimodal_blocks, errors
