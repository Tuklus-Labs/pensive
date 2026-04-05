"""Email JSONL parser for spreading activation ingestion.

Parses pre-processed email JSONL files (one JSON object per line) with fields:
    from, to, subject, date, body

Designed for the mbox_parser output at ~/dell_backup/Gary/Downloads/mbox_parser/
but works with any JSONL following that schema.
"""
import json
import logging
from pathlib import Path
from typing import Iterator
from hashlib import sha256

from ..base import SADocument, BaseParser
from ..chunker import SentenceAwareChunker
from ..query_gen import QueryGenerator

logger = logging.getLogger(__name__)

# Module-level singletons -- both classes are stateless, no need to re-instantiate per parser.
_DEFAULT_CHUNKER = SentenceAwareChunker()
_DEFAULT_QUERY_GEN = QueryGenerator()


class EmailJSONLParser(BaseParser):
    """Parse email JSONL files into SADocuments."""

    def __init__(self, jsonl_path: str, chunker: SentenceAwareChunker = None):
        """Initialize the email parser.

        Args:
            jsonl_path: Path to a .jsonl file or directory containing .jsonl files.
            chunker: Optional chunker instance. Defaults to SentenceAwareChunker().
        """
        self.path = Path(jsonl_path)
        self.chunker = chunker or _DEFAULT_CHUNKER
        self.query_gen = _DEFAULT_QUERY_GEN

    def source_name(self) -> str:
        return 'email'

    def _jsonl_files(self) -> list[Path]:
        """Find all JSONL files to process."""
        if self.path.is_file() and self.path.suffix == '.jsonl':
            return [self.path]
        if self.path.is_dir():
            return sorted(self.path.glob('*.jsonl'))
        return []

    def parse(self) -> Iterator[SADocument]:
        """Yield SADocuments from email JSONL files."""
        files = self._jsonl_files()
        if not files:
            logger.warning("No JSONL files found at: %s", self.path)
            return

        total = 0
        for jsonl_file in files:
            logger.info("Parsing %s", jsonl_file.name)
            count = 0
            with open(jsonl_file, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        email = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed JSON at %s:%d", jsonl_file.name, line_num)
                        continue

                    sender = email.get('from', '')
                    recipient = email.get('to', '')
                    subject = email.get('subject', '(no subject)')
                    date = email.get('date', '')
                    body = email.get('body', '')

                    if not body or len(body.strip()) < 20:
                        continue

                    # Build document text
                    header = f"From: {sender}\nTo: {recipient}\nSubject: {subject}\nDate: {date}"
                    full_text = f"{header}\n\n{body}"

                    # Generate stable doc ID
                    doc_id = sha256(f"{date}:{sender}:{subject}".encode()).hexdigest()[:16]

                    # Chunk long emails
                    chunks = self.chunker.chunk(full_text) if len(full_text) > 1200 else [full_text]

                    for i, chunk in enumerate(chunks):
                        chunk_id = f"email-{doc_id}-{i}" if len(chunks) > 1 else f"email-{doc_id}"

                        # Value is a summary line for search results
                        value = f"[{date[:10] if date else '?'}] {sender} -> {recipient}: {subject}"
                        if len(chunks) > 1:
                            value += f" (part {i+1}/{len(chunks)})"

                        query = self.query_gen.generate(chunk) if hasattr(self.query_gen, 'generate') else ''

                        yield SADocument(
                            content=chunk,
                            doc_id=chunk_id,
                            value=value,
                            query=query,
                            source='email',
                            metadata={
                                'from': sender,
                                'to': recipient,
                                'subject': subject,
                                'date': date,
                                'file': jsonl_file.name,
                            },
                        )
                        count += 1

            total += count
            logger.info("  %s: %d documents", jsonl_file.name, count)

        logger.info("Email total: %d documents from %d files", total, len(files))
