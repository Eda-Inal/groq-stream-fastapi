"""
Recursive text chunking with tiktoken token limits and overlap.

Splitting order for oversized segments: paragraphs → lines → sentences → token windows.
Code fences (``` ... ```) are kept intact when possible; oversized blocks split on newlines only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

import tiktoken

from app.core.chunking_config import TIKTOKEN_ENCODING
from app.core.config import settings

_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens; returns 0 on empty string. Avoids raising on odd input."""
    if not text:
        return 0
    try:
        return len(_get_encoding().encode(text))
    except Exception:
        return 0


class DocumentTooLargeError(ValueError):
    """Raised when document exceeds MAX_DOCUMENT_TOKENS after normalization."""


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk ready for embedding / storage."""

    # — Identity —
    chunk_id: str
    doc_id: int
    chunk_index: int
    total_chunks: int  # placeholder 0 until ingestion_service post-processes
    # — Content —
    text: str
    token_count: int
    # — Source —
    source_filename: str
    page_number: int | None
    # — Structure —
    section_heading: str | None
    context_prefix: str  # reserved for future use, always "" for now


_FENCE_RE = re.compile(r"(```(?:[a-zA-Z0-9_-]*)?\n?)([\s\S]*?)(```)", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _make_context_prefix(source_filename: str, page_number: int | None) -> str:
    if source_filename and page_number is not None:
        return f"[Belge: {source_filename} | Sayfa: {page_number}]"
    if source_filename:
        return f"[Belge: {source_filename}]"
    return ""
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:\d+[.)]\s+|[-*•]\s+)")


def normalize_document_text(text: str) -> str:
    """
    UTF-8 safe string, strip NULs, normalize newlines, trim runaway blank lines.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^[ \t]*---[ \t]*$", "", text)
    # Collapse 3+ newlines to double newline (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_markdown_table_line(line: str) -> bool:
    s = line.strip()
    if not s or "|" not in s:
        return False
    # Heuristic: table row or separator
    return s.startswith("|") or s.lstrip().startswith("|")


def _split_table_runs(text: str) -> list[tuple[str, str]]:
    """
    Split `text` into alternating ('plain'|'table', segment) pieces.
    Consecutive markdown-style table lines form one table run.
    """
    lines = text.split("\n")
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    mode: str | None = None

    def flush() -> None:
        nonlocal buf, mode
        if buf and mode:
            parts.append((mode, "\n".join(buf)))
        buf = []
        mode = None

    for line in lines:
        is_tbl = _is_markdown_table_line(line)
        m = "table" if is_tbl else "plain"
        if mode is None:
            mode = m
            buf = [line]
        elif m == mode:
            buf.append(line)
        else:
            flush()
            mode = m
            buf = [line]
    flush()
    return parts if parts else [("plain", text)]


def _extract_list_segments(text: str) -> list[tuple[str, str]]:
    """
    Split a plain-text segment into ('prose', block) and ('list_block', block) parts.

    Consecutive list items (numbered or bulleted) are grouped into a single
    ('list_block', ...) entry so they share one embedding vector with full
    context instead of becoming many tiny individual chunks.
    Non-list lines are grouped into ('prose', ...) blocks as before.
    """
    lines = text.split("\n")
    parts: list[tuple[str, str]] = []
    prose_buf: list[str] = []
    list_buf: list[str] = []

    def _flush_prose() -> None:
        if prose_buf:
            block = "\n".join(prose_buf).strip()
            if block:
                parts.append(("prose", block))
            prose_buf.clear()

    def _flush_list() -> None:
        if list_buf:
            block = "\n".join(list_buf).strip()
            if block:
                parts.append(("list_block", block))
            list_buf.clear()

    for line in lines:
        if _LIST_ITEM_RE.match(line) and line.strip():
            _flush_prose()
            list_buf.append(line.strip())
        else:
            _flush_list()
            prose_buf.append(line)

    _flush_prose()
    _flush_list()
    return parts or [("prose", text)]


def _split_by_markdown_headings(text: str) -> list[tuple[str, str, str | None]]:
    """
    Split plain text on markdown headings (# through ######).
    Returns list of ('section', body_text, heading_title | None) tuples.
    heading_title is the heading text without # marks (e.g. "Introduction").
    Heading text is prepended to its section body so downstream chunks carry context.
    If no headings found, returns [('section', text, None)].
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("section", text, None)]

    parts: list[tuple[str, str, str | None]] = []
    # Text before the first heading (if any)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        parts.append(("section", preamble, None))

    pending_prefix = ""

    for i, m in enumerate(matches):
        heading_line = m.group(0)   # e.g. "## **Introduction**"
        # Strip inline markdown formatting (**, *, __, _, `) from the metadata
        # field — PyMuPDF markdown mode may wrap bold headings in ** markers.
        heading_title = re.sub(r"[*_`]", "", m.group(2)).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if body:
            if pending_prefix:
                section = (pending_prefix + "\n\n" + heading_line + "\n\n" + body).strip()
            else:
                section = (heading_line + "\n\n" + body).strip()
            parts.append(("section", section, heading_title))
            pending_prefix = ""
        else:
            pending_prefix = (
                (pending_prefix + "\n\n" + heading_line).strip()
                if pending_prefix else heading_line
            )

    if pending_prefix:
        parts.append(("section", pending_prefix, None))

    return parts


def _split_by_code_fences(text: str) -> list[tuple[str, str]]:
    """
    Split into ('code', inner) and ('plain', body) segments in order.
    Inner code does not include the fence lines themselves.
    """
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
        before = text[pos : m.start()]
        if before:
            out.append(("plain", before))
        opener, inner, closer = m.group(1), m.group(2), m.group(3)
        # Reconstruct fenced block as atomic content for downstream (keep fences for context)
        fenced = f"{opener}{inner}{closer}"
        out.append(("code", fenced))
        pos = m.end()
    tail = text[pos:]
    if tail:
        out.append(("plain", tail))
    if not out:
        out.append(("plain", text))
    return out


def _token_window_chunks(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Last resort: fixed token windows with overlap."""
    enc = _get_encoding()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return [text] if text else []

    out: list[str] = []
    start = 0
    step = max(1, max_tokens - overlap)
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        piece = enc.decode(ids[start:end])
        if piece.strip():
            out.append(piece)
        if end >= len(ids):
            break
        start += step
    return out


def _split_oversized_segment(segment: str, max_tokens: int, overlap: int) -> list[str]:
    """Apply paragraph → line → sentence → token window for one segment."""
    if count_tokens(segment) <= max_tokens:
        return [segment] if segment else []

    # 1) Paragraphs — use a tighter effective limit so that each paragraph
    #    stays in its own topical chunk.  Merging two paragraphs is only
    #    allowed when the combined result is well under max_tokens; this
    #    prevents unrelated topics from sharing the same embedding vector.
    #    Effective limit = max_tokens minus the overlap budget (the next
    #    chunk will re-read that many tokens from us anyway).
    para_merge_limit = max(max_tokens - overlap, (max_tokens * 2) // 3)
    paras = re.split(r"\n\s*\n", segment)
    if len(paras) > 1:
        merged = _merge_parts_under_limit(paras, "\n\n", para_merge_limit, overlap)
        if all(count_tokens(x) <= max_tokens for x in merged):
            return merged

    # 2) Lines
    lines = segment.split("\n")
    if len(lines) > 1:
        merged = _merge_parts_under_limit(lines, "\n", max_tokens, overlap)
        if all(count_tokens(x) <= max_tokens for x in merged):
            return merged

    # 3) Sentences (. )
    pieces = re.split(r"(?<=\.)\s+", segment)
    if len(pieces) > 1:
        merged = _merge_parts_under_limit(pieces, " ", max_tokens, overlap)
        if all(count_tokens(x) <= max_tokens for x in merged):
            return merged

    return _token_window_chunks(segment, max_tokens, overlap)


def _overlap_prefix_text(previous_chunk: str, overlap_tokens: int) -> str:
    """Return the last complete sentence(s) from previous_chunk whose total
    tokens fit within overlap_tokens.  Falls back to the full chunk when it is
    already small enough, and never returns a raw mid-sentence token slice."""
    if overlap_tokens <= 0 or not previous_chunk:
        return ""
    if count_tokens(previous_chunk) <= overlap_tokens:
        return previous_chunk
    sentences = re.split(r"(?<=[.!?])\s+", previous_chunk.strip())
    selected: list[str] = []
    total = 0
    for sent in reversed(sentences):
        t = count_tokens(sent)
        if total + t <= overlap_tokens:
            selected.insert(0, sent)
            total += t
        else:
            break
    return " ".join(selected) if selected else ""


def _merge_parts_under_limit(
    parts: list[str],
    joiner: str,
    max_tokens: int,
    overlap: int,
) -> list[str]:
    """Greedy merge of small parts into chunks <= max_tokens; prepend overlap from previous."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = joiner.join(current).strip()
        if body:
            chunks.append(body)
        current = []
        current_tokens = 0

    for p in parts:
        p = p.strip()
        if not p:
            continue
        pt = count_tokens(p)
        if pt > max_tokens:
            flush()
            for sub in _split_oversized_segment(p, max_tokens, overlap):
                if chunks and overlap > 0:
                    prefix = _overlap_prefix_text(chunks[-1], overlap)
                    if prefix and not sub.startswith(prefix):
                        sub = (prefix + joiner + sub).strip()
                if count_tokens(sub) > max_tokens:
                    for win in _token_window_chunks(sub, max_tokens, overlap):
                        chunks.append(win)
                else:
                    chunks.append(sub)
            current = []
            current_tokens = 0
            continue

        added = pt + (count_tokens(joiner) if current else 0)
        if current_tokens + added <= max_tokens:
            current.append(p)
            current_tokens += added
        else:
            flush()
            if chunks and overlap > 0:
                prefix = _overlap_prefix_text(chunks[-1], overlap)
                if prefix:
                    current = [prefix, p]
                    current_tokens = count_tokens(joiner.join(current))
                else:
                    current = [p]
                    current_tokens = pt
            else:
                current = [p]
                current_tokens = pt

    flush()
    return [c for c in chunks if c]


def _chunk_plain_text(
    text: str,
    max_tokens: int,
    overlap: int,
) -> list[tuple[str, str | None]]:
    """Plain text: heading split → table runs → list-block grouping → recursive split.
    Returns (chunk_text, detected_heading | None) pairs."""
    text = text.strip()
    if not text:
        return []

    result: list[tuple[str, str | None]] = []
    for _, section, heading in _split_by_markdown_headings(text):
        section_chunks: list[str] = []
        runs = _split_table_runs(section)
        for kind, seg in runs:
            if kind == "table":
                if count_tokens(seg) <= max_tokens:
                    section_chunks.append(seg)
                else:
                    lines = seg.split("\n")
                    section_chunks.extend(_merge_parts_under_limit(lines, "\n", max_tokens, overlap))
            else:
                for lkind, lseg in _extract_list_segments(seg):
                    if lkind == "list_block":
                        if count_tokens(lseg) <= max_tokens:
                            section_chunks.append(lseg)
                        else:
                            items = lseg.split("\n")
                            section_chunks.extend(
                                _merge_parts_under_limit(items, "\n", max_tokens, overlap)
                            )
                    else:
                        section_chunks.extend(_split_oversized_segment(lseg, max_tokens, overlap))

        for c in section_chunks:
            if c.strip():
                result.append((c, heading))

    return result


def _chunk_code_fenced(block: str, max_tokens: int, overlap: int) -> list[str]:
    """Keep fence block intact if it fits; else split inner lines only."""
    if count_tokens(block) <= max_tokens:
        return [block]

    m = _FENCE_RE.search(block)
    if not m:
        return _split_oversized_segment(block, max_tokens, overlap)

    opener, inner, closer = m.group(1), m.group(2), m.group(3)
    inner_chunks = _merge_parts_under_limit(inner.split("\n"), "\n", max_tokens - 20, overlap)
    out: list[str] = []
    for ic in inner_chunks:
        out.append(f"{opener}{ic}{closer}")
    return out


def _drop_tiny_chunks(chunks: list[ChunkRecord], min_tokens: int) -> list[ChunkRecord]:
    """
    Drop noise fragments from any position.

    A chunk is kept if ANY of these is true:
    - token_count >= min_tokens  (normal sized content)
    - starts with ``` (code fence — often legitimately short)
    - token_count >= 5 AND contains at least one letter  (short but real text,
      e.g. a section heading with a brief intro sentence)

    The letter check removes PDF noise: standalone bullet symbols (•),
    page-number footers ("5"), and separator lines with no words.

    If filtering would remove everything, returns the original list unchanged.
    """
    if len(chunks) <= 1:
        return chunks
    filtered = [
        c for c in chunks
        if c.token_count >= min_tokens
        or c.text.lstrip().startswith("```")
        or (c.token_count >= 5 and any(ch.isalpha() for ch in c.text))
    ]
    return filtered if filtered else chunks


def chunk_document(
    text: str,
    *,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
    max_document_tokens: int | None = None,
    short_doc_max_tokens: int | None = None,
    min_chunk_tokens: int | None = None,
    page_number: int | None = None,
    section_heading: str | None = None,
    doc_id: int = 0,
    source_filename: str = "",
) -> list[ChunkRecord]:
    """
    Normalize and split document into chunks with metadata.

    Omitted numeric arguments use `settings` (env / .env). See `.env.example`.

    Raises DocumentTooLargeError if token count exceeds max_document_tokens.
    """
    cst = chunk_size_tokens if chunk_size_tokens is not None else settings.chunk_size_tokens
    ovt = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens
    mdt = max_document_tokens if max_document_tokens is not None else settings.max_document_tokens
    sdm = (
        short_doc_max_tokens
        if short_doc_max_tokens is not None
        else settings.short_doc_single_chunk_max_tokens
    )
    mct = min_chunk_tokens if min_chunk_tokens is not None else settings.min_chunk_tokens

    normalized = normalize_document_text(text)
    total = count_tokens(normalized)
    if total > mdt:
        raise DocumentTooLargeError(
            f"Document exceeds maximum length ({mdt} tokens); got {total}."
        )

    if total == 0:
        return []

    if total <= sdm:
        return [
            ChunkRecord(
                chunk_id=str(uuid4()),
                doc_id=doc_id,
                chunk_index=0,
                total_chunks=0,
                text=normalized,
                token_count=total,
                source_filename=source_filename,
                page_number=page_number,
                section_heading=section_heading,
                context_prefix=_make_context_prefix(source_filename, page_number),
            )
        ]

    pieces = _split_by_code_fences(normalized)
    raw_chunks: list[tuple[str, str | None]] = []
    for kind, seg in pieces:
        if kind == "code":
            raw_chunks.extend((c, None) for c in _chunk_code_fenced(seg, cst, ovt))
        else:
            raw_chunks.extend(_chunk_plain_text(seg, cst, ovt))

    _prefix = _make_context_prefix(source_filename, page_number)
    records = [
        ChunkRecord(
            chunk_id=str(uuid4()),
            doc_id=doc_id,
            chunk_index=i,
            total_chunks=0,
            text=c.strip(),
            token_count=count_tokens(c.strip()),
            source_filename=source_filename,
            page_number=page_number,
            section_heading=detected_heading or section_heading,
            context_prefix=_prefix,
        )
        for i, (c, detected_heading) in enumerate(
            (c, h) for c, h in raw_chunks if c.strip()
        )
    ]
    records = _drop_tiny_chunks(records, mct)
    return records
