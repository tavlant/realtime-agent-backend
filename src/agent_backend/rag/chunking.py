"""Markdown-aware text chunking.

Documents are first split into sections by Markdown headings, so a chunk never straddles
two topics, and each chunk is prefixed with its heading path ("Guide > Install") so the
embedding carries that context. Long sections are split recursively on progressively
finer separators (paragraphs, lines, sentences, words) and packed into chunks of at most
`max_chars`, with a character overlap between neighbours to keep boundary context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    heading: str | None = None


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown into (heading_path, body) sections."""
    sections: list[tuple[str | None, str]] = []
    path: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            heading = " > ".join(title for _, title in path) or None
            sections.append((heading, body))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = [(lvl, title) for lvl, title in path if lvl < level]
            path.append((level, match.group(2)))
        else:
            buffer.append(line)
    flush()
    return sections


def _split_recursive(text: str, max_chars: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    sep, rest = separators[0], separators[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) <= 1:
        return _split_recursive(text, max_chars, rest)

    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            out.append(current)
        if len(piece) > max_chars:
            out.extend(_split_recursive(piece, max_chars, rest))
            current = ""
        else:
            current = piece
    if current:
        out.append(current)
    return out


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        # Keep the heading prefix bounded so every chunk respects max_chars.
        prefix = f"{heading[: max_chars // 4]}\n" if heading else ""
        budget = max_chars - len(prefix)
        piece_limit = max(budget - overlap - 1, 1) if overlap else budget
        pieces = _split_recursive(body, piece_limit, _SEPARATORS)
        previous = ""
        for piece in pieces:
            carried = _overlap_tail(previous, overlap)
            content = f"{carried} {piece}".strip() if carried else piece.strip()
            chunks.append(Chunk(index=len(chunks), text=f"{prefix}{content}", heading=heading))
            previous = piece
    return chunks
