"""
Recursive text chunking with tiktoken token limits and overlap.

Splitting order for oversized segments: paragraphs → lines → sentences → token windows.
Code fences (``` ... ```) are kept intact when possible; oversized blocks split on newlines only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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

    text: str
    token_count: int
    page_number: int | None = None
    section_heading: str | None = None


_FENCE_RE = re.compile(r"(```(?:[a-zA-Z0-9_-]*)?\n?)([\s\S]*?)(```)", re.MULTILINE)


def normalize_document_text(text: str) -> str:
    """
    UTF-8 safe string, strip NULs, normalize newlines, trim runaway blank lines.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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

    # 1) Paragraphs
    paras = re.split(r"\n\s*\n", segment)
    if len(paras) > 1:
        merged = _merge_parts_under_limit(paras, "\n\n", max_tokens, overlap)
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
) -> list[str]:
    """Plain text: table runs + recursive split."""
    text = text.strip()
    if not text:
        return []

    runs = _split_table_runs(text)
    all_chunks: list[str] = []
    for kind, seg in runs:
        if kind == "table":
            # Atomic when possible; if huge, split lines only
            if count_tokens(seg) <= max_tokens:
                all_chunks.append(seg)
            else:
                lines = seg.split("\n")
                all_chunks.extend(_merge_parts_under_limit(lines, "\n", max_tokens, overlap))
        else:
            all_chunks.extend(_split_oversized_segment(seg, max_tokens, overlap))

    return [c for c in all_chunks if c.strip()]


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
    Drop only tiny *trailing* prose fragments.

    Does not strip trailing markdown code fences (they often fall below MIN_CHUNK_TOKENS).
    """
    if len(chunks) <= 1:
        return chunks
    out = list(chunks)
    while len(out) > 1 and out[-1].token_count < min_tokens:
        if out[-1].text.lstrip().startswith("```"):
            break
        out.pop()
    return out


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
                text=normalized,
                token_count=total,
                page_number=page_number,
                section_heading=section_heading,
            )
        ]

    pieces = _split_by_code_fences(normalized)
    raw_chunks: list[str] = []
    for kind, seg in pieces:
        if kind == "code":
            raw_chunks.extend(_chunk_code_fenced(seg, cst, ovt))
        else:
            raw_chunks.extend(_chunk_plain_text(seg, cst, ovt))

    records = [
        ChunkRecord(
            text=c.strip(),
            token_count=count_tokens(c.strip()),
            page_number=page_number,
            section_heading=section_heading,
        )
        for c in raw_chunks
        if c.strip()
    ]
    records = _drop_tiny_chunks(records, mct)
    return records
