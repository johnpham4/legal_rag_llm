import re
from typing import List
from loguru import logger


def chunk_legal_document(
    text: str,
    min_length: int = 100,
    max_length: int = 1000,
) -> List[str]:
    """
    Chunk Vietnamese legal documents by legal structure.

    Handles 3 types of legal documents:
    1. Full structure: Chương → Điều → Khoản → Điểm
    2. No Chương: Điều → Khoản → Điểm
    3. No Chương & Điều: Khoản → Điểm only

    Args:
        text: Cleaned legal document text
        min_length: Minimum chars for preamble to be saved as separate chunk
        max_length: Maximum characters per chunk

    Returns:
        List of text chunks preserving legal structure
    """
    if not text:
        return []

    chunks = []

    # Detect document structure
    has_chuong = bool(re.search(r'Chương\s+[IVXLCDM\d]+:', text, re.IGNORECASE))
    has_dieu = bool(re.search(r'Điều\s+\d+:', text))
    has_khoan = bool(re.search(r'Khoản\s+\d+\.', text))

    # Extract preamble (header before main content)
    # Strategy: Look for "QUYẾT ĐỊNH:" (2nd occurrence) or "NGHỊ QUYẾT:" (with colon)
    # These mark the start of actual legal content
    preamble = ""
    content_start = 0

    # Try to find "QUYẾT ĐỊNH:" (find 2nd occurrence if exists)
    quyet_dinh_pattern = r'QUYẾT ĐỊNH\s*:'
    matches = list(re.finditer(quyet_dinh_pattern, text, re.IGNORECASE))

    if len(matches) >= 2:
        # Has 2+ "QUYẾT ĐỊNH:" - use 2nd one as content start
        preamble = text[:matches[1].start()].strip()
        content_start = matches[1].end()
    elif len(matches) == 1:
        # Only 1 "QUYẾT ĐỊNH:" - use it as content start
        preamble = text[:matches[0].start()].strip()
        content_start = matches[0].end()
    else:
        # Try "NGHỊ QUYẾT:" pattern
        nghi_quyet_pattern = r'NGHỊ QUYẾT\s*:'
        match = re.search(nghi_quyet_pattern, text, re.IGNORECASE)
        if match:
            preamble = text[:match.start()].strip()
            content_start = match.end()
        else:
            # Fallback: use Chương/Điều/Khoản patterns (old logic)
            preamble_patterns = [
                r'(Chương\s+[IVXLCDM\d]+:)',
                r'(Điều\s+\d+:)',
                r'(Khoản\s+\d+\.)'
            ]
            for pattern in preamble_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    preamble = text[:match.start()].strip()
                    content_start = match.start()
                    break

    # Only save preamble if it's long enough to contain meaningful info
    if preamble and len(preamble) >= min_length:
        chunks.append(preamble)

    main_content = text[content_start:] if content_start > 0 else text

    # Route to appropriate chunking strategy
    if has_chuong and has_dieu:
        # logger.debug("Chunking strategy: Chương → Điều → Khoản")
        content_chunks = _chunk_with_chuong(main_content, max_length)
    elif has_dieu:

        content_chunks = _chunk_by_dieu(main_content, max_length)
    elif has_khoan:
        # logger.debug("Chunking strategy: Khoản → Điểm")
        content_chunks = _chunk_by_khoan_only(main_content, max_length)
    else:
        # logger.warning("No legal structure detected, using fallback chunking")
        content_chunks = _chunk_by_size(main_content, max_length)

    chunks.extend(content_chunks)
    return [chunk for chunk in chunks if chunk.strip()]


def _chunk_with_chuong(text: str, max_size: int) -> List[str]:
    """
    Case 1: Document has Chương → Điều → Khoản → Điểm
    Strategy: Keep Chương header with each Điều chunk
    """
    chuong_pattern = r'(Chương\s+[IVXLCDM\d]+:[^\n]*)'
    parts = re.split(chuong_pattern, text, flags=re.IGNORECASE)

    chunks = []
    current_chuong = ""

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        if re.match(chuong_pattern, part, re.IGNORECASE):
            current_chuong = part
        else:
            # Content under this Chương - split by Điều
            dieu_chunks = _chunk_by_dieu(part, max_size, chuong_header=current_chuong)
            chunks.extend(dieu_chunks)

    return chunks


def _has_muc(text: str) -> bool:
    """Check if text contains Mục (Roman numeral sections like I., II., III.)"""
    return bool(re.search(r'\n(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.\s+', text))


def _chunk_by_dieu(text: str, max_size: int, chuong_header: str = "") -> List[str]:
    """
    Case 2: Split by Điều (with or without Chương)
    Strategy: Each Điều = 1 chunk (if fits), otherwise split by Khoản or Mục
    """
    dieu_pattern = r'(Điều\s+\d+:[^\n]*)'
    parts = re.split(dieu_pattern, text)

    chunks = []

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        if not part:
            i += 1
            continue

        if re.match(dieu_pattern, part):
            # This is "Điều X: Title"
            if i + 1 < len(parts):
                dieu_header = part
                dieu_content = parts[i + 1].strip()

                # Build full chunk with Chương context if available
                full_text = ""
                if chuong_header:
                    full_text = f"{chuong_header}\n\n"
                full_text += f"{dieu_header}\n{dieu_content}"

                # Check if fits in one chunk
                if len(full_text) <= max_size:
                    chunks.append(full_text)
                else:
                    # Too long, check if has Mục (Roman numerals) first
                    header = f"{chuong_header}\n{dieu_header}" if chuong_header else dieu_header

                    if _has_muc(dieu_content):
                        # Split by Mục (I., II., III., etc)
                        sub_chunks = _chunk_by_muc(dieu_content, max_size, header)
                    else:
                        # Split by Khoản
                        sub_chunks = _chunk_by_khoan(dieu_content, max_size, header)

                    chunks.extend(sub_chunks)

                i += 2
            else:
                i += 1
        else:
            # Content without Điều (intro or preamble within section)
            if len(part) > 50:
                if chuong_header:
                    part = f"{chuong_header}\n\n{part}"
                chunks.append(part[:max_size] if len(part) > max_size else part)
            i += 1

    return chunks


def _chunk_by_khoan_only(text: str, max_size: int) -> List[str]:
    """
    Case 3: Document has no Chương and no Điều, only Khoản → Điểm
    Strategy: Group multiple Khoản into chunks
    """
    khoan_pattern = r'(Khoản\s+\d+\.)'
    parts = re.split(khoan_pattern, text)

    chunks = []
    current_chunk = ""

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        if not part:
            i += 1
            continue

        if re.match(khoan_pattern, part):
            # This is "Khoản X."
            if i + 1 < len(parts):
                khoan_text = part + " " + parts[i + 1].strip()

                # Try to add to current chunk
                if not current_chunk:
                    current_chunk = khoan_text
                elif len(current_chunk) + len(khoan_text) + 2 <= max_size:
                    current_chunk += "\n\n" + khoan_text
                else:
                    # Current chunk is full, save it and start new
                    chunks.append(current_chunk)
                    current_chunk = khoan_text

                i += 2
            else:
                i += 1
        else:
            # Content before first Khoản
            if not current_chunk and len(part) > 30:
                current_chunk = part[:max_size] if len(part) > max_size else part
            i += 1

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _chunk_by_muc(content: str, max_size: int, header: str) -> List[str]:
    """
    Split Điều content by Mục (Roman numerals: I., II., III., etc).
    Each Mục section becomes a separate chunk with the Điều header.

    Args:
        content: Content of the Điều (without header)
        max_size: Maximum chunk size
        header: Điều header (and possibly Chương) to prepend

    Returns:
        List of chunks, each with header + one Mục section
    """
    muc_pattern = r'\n(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.\s+([^\n]+)'
    parts = re.split(muc_pattern, content)

    chunks = []

    # First part is intro text before first Mục (if any)
    intro = parts[0].strip()
    if intro and len(intro) > 30:
        chunks.append(f"{header}\n{intro}")

    # Process Mục sections (parts come in groups of 3: roman, title, content)
    i = 1
    while i < len(parts):
        if i + 2 < len(parts):
            roman = parts[i]
            muc_title = parts[i + 1].strip()
            muc_content = parts[i + 2].strip()

            # Build chunk: header + "I. Title" + content
            muc_text = f"{roman}. {muc_title}"
            if muc_content:
                muc_text += f"\n{muc_content}"

            full_chunk = f"{header}\n{muc_text}"

            # If still too large, split by Khoản within this Mục
            if len(full_chunk) > max_size:
                muc_header = f"{header}\n{roman}. {muc_title}"
                sub_chunks = _chunk_by_khoan(muc_content, max_size, muc_header)
                chunks.extend(sub_chunks)
            else:
                chunks.append(full_chunk)

            i += 3
        else:
            break

    return chunks


def _chunk_by_khoan(content: str, max_size: int, header: str) -> List[str]:
    """
    Split Điều content by Khoản when Điều is too long.
    Each chunk keeps the Điều header for context.

    Args:
        content: Content of the Điều (without header)
        max_size: Maximum chunk size
        header: Điều header (and possibly Chương) to prepend

    Returns:
        List of chunks, each with header + some Khoản
    """
    khoan_pattern = r'(Khoản\s+\d+\.)'
    parts = re.split(khoan_pattern, content)

    chunks = []
    current_chunk = header

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        if not part:
            i += 1
            continue

        if re.match(khoan_pattern, part):
            # This is "Khoản X."
            if i + 1 < len(parts):
                khoan_text = part + " " + parts[i + 1].strip()

                # Check if we can add this Khoản to current chunk
                test_chunk = current_chunk + "\n" + khoan_text

                if len(test_chunk) <= max_size:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, save it
                    if current_chunk.strip() != header.strip():
                        chunks.append(current_chunk.strip())
                    # Start new chunk with header + this Khoản
                    current_chunk = header + "\n" + khoan_text

                i += 2
            else:
                i += 1
        else:
            # Intro text before first Khoản
            if i == 0 and len(part) > 20:
                current_chunk += "\n" + part
            i += 1

    # Add final chunk
    if current_chunk.strip() != header.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _chunk_by_size(text: str, max_size: int) -> List[str]:
    """
    Fallback chunking when no legal structure is detected.
    Split by sentences and paragraphs.
    """
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len(para) + 2 <= max_size:
            current_chunk += "\n\n" + para
        else:
            chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# def chunk_with_embeddings(
#     text: str,
#     chunk_size: int = 500,
#     chunk_overlap: int = 50
# ) -> List[str]:
#     """
#     Chunk text based on embedding model's token limits.
#     Fallback method for non-legal documents.
#     """
#     from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

#     embedding_model = EmbeddingModelSingleton()

#     # First split by paragraphs
#     character_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ". ", " "],
#         chunk_size=chunk_size,
#         chunk_overlap=0
#     )
#     text_split = character_splitter.split_text(text)

#     # Then split by tokens
#     token_splitter = SentenceTransformersTokenTextSplitter(
#         chunk_overlap=chunk_overlap,
#         tokens_per_chunk=embedding_model.max_input_length,
#         model_name=embedding_model.model_id,
#     )

#     chunks = []
#     for section in text_split:
#         chunks.extend(token_splitter.split_text(section))

#     return chunks
