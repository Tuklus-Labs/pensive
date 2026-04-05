"""Sentence-aware text chunking for the ingestion pipeline."""
import re
from typing import List


_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')


class SentenceAwareChunker:
    """Split text into chunks on sentence boundaries with overlap."""

    def __init__(self, target_size: int = 800, overlap: int = 100, min_size: int = 50):
        self.target_size = target_size
        self.overlap = overlap
        self.min_size = min_size

    def chunk(self, text: str) -> List[str]:
        """Split text into sentence-aware chunks.

        - Returns [text] if the whole thing fits in target_size.
        - Splits on sentence-ending punctuation or double newlines.
        - Keeps an overlap of the last few sentences (up to overlap chars)
          at the start of each new chunk for continuity.
        - If the final chunk is too short (< min_size), it gets appended
          to the previous chunk instead of standing alone.
        """
        if len(text) <= self.target_size:
            return [text]

        sentences = _SENTENCE_RE.split(text)
        # Drop empty fragments from the split
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            added_len = len(sentence) + (1 if current_sentences else 0)  # space separator

            if current_sentences and current_len + added_len > self.target_size:
                # Flush current chunk
                chunks.append(' '.join(current_sentences))

                # Build overlap: take trailing sentences up to overlap chars
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + 1 > self.overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1

                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences) + max(0, len(current_sentences) - 1)

            current_sentences.append(sentence)
            current_len += added_len

        # Handle the final chunk
        if current_sentences:
            final_text = ' '.join(current_sentences)
            if len(final_text) < self.min_size and chunks:
                # Too short -- append to previous chunk
                chunks[-1] = chunks[-1] + ' ' + final_text
            else:
                chunks.append(final_text)

        return chunks
