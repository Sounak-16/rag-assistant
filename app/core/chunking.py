import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    page: Optional[int] = None


class TextChunker:
    """Boundary-aware text chunker that preserves headings, paragraphs, and sentence boundaries."""

    SEPARATORS = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        " ",
        "",
    ]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Chunk]:
        """Split a string of text into structured chunks with overlap and boundary preservation."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        raw_chunks = self._recursive_split(cleaned_text, self.SEPARATORS)
        merged_chunks = self._merge_with_overlap(raw_chunks, cleaned_text)

        chunks: List[Chunk] = []
        for i, (chunk_text, start_char, end_char) in enumerate(merged_chunks):
            chunk_id = f"{document_id}_p{page if page is not None else 0}_c{start_index + i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    text=chunk_text,
                    chunk_index=start_index + i,
                    char_start=start_char,
                    char_end=end_char,
                    page=page,
                )
            )

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first matching separator."""
        if len(text) <= self.chunk_size or not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level fallback
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        if separator not in text:
            return self._recursive_split(text, remaining_separators)

        splits = text.split(separator)
        result: List[str] = []

        for i, part in enumerate(splits):
            if not part:
                continue
            # Re-attach separator when reconstructing
            formatted_part = part if i == 0 else separator.lstrip("\n") + part
            if len(formatted_part) > self.chunk_size:
                result.extend(self._recursive_split(formatted_part, remaining_separators))
            else:
                result.append(formatted_part)

        return result

    def _merge_with_overlap(self, pieces: List[str], full_text: str) -> List[tuple[str, int, int]]:
        """Greedily assemble pieces up to chunk_size, applying overlap across chunks."""
        chunks: List[tuple[str, int, int]] = []
        current_pieces: List[str] = []
        current_length = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_length + piece_len > self.chunk_size and current_pieces:
                chunk_str = "".join(current_pieces).strip()
                if chunk_str:
                    start_char = full_text.find(chunk_str)
                    end_char = start_char + len(chunk_str) if start_char != -1 else len(chunk_str)
                    chunks.append((chunk_str, max(0, start_char), end_char))

                # Build overlap from the tail of current pieces
                overlap_pieces: List[str] = []
                overlap_len = 0
                for p in reversed(current_pieces):
                    if overlap_len + len(p) <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break

                current_pieces = overlap_pieces + [piece]
                current_length = sum(len(p) for p in current_pieces)
            else:
                current_pieces.append(piece)
                current_length += piece_len

        if current_pieces:
            chunk_str = "".join(current_pieces).strip()
            if chunk_str:
                start_char = full_text.find(chunk_str)
                end_char = start_char + len(chunk_str) if start_char != -1 else len(chunk_str)
                chunks.append((chunk_str, max(0, start_char), end_char))

        return chunks
