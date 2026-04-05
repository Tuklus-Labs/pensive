"""Facebook Messenger export parser for spreading activation ingestion.

Parses Facebook data exports (JSON format) from:
    {export_dir}/your_facebook_activity/messages/

Handles inbox/, archived_threads/, message_requests/, and e2ee_cutover/
subdirectories. Each thread directory contains a message_1.json with all
messages for that conversation.

Facebook exports encode text with Latin-1 mojibake (UTF-8 bytes stored as
Latin-1 codepoints). The _fix_encoding() method reverses this.
"""
import json
import logging
from pathlib import Path
from typing import Iterator

from ..base import SADocument, BaseParser
from ..chunker import SentenceAwareChunker
from ..query_gen import QueryGenerator

logger = logging.getLogger(__name__)

# Module-level singletons -- both classes are stateless, no need to re-instantiate per parser.
_DEFAULT_CHUNKER = SentenceAwareChunker()
_DEFAULT_QUERY_GEN = QueryGenerator()

# Subdirectories under your_facebook_activity/messages/ that contain threads
THREAD_SUBDIRS = ('inbox', 'archived_threads', 'message_requests', 'e2ee_cutover')


class FacebookParser(BaseParser):
    """Parse Facebook Messenger exports into SADocuments."""

    def __init__(self, export_dir: str, chunker: SentenceAwareChunker = None):
        """Initialize the Facebook parser.

        Args:
            export_dir: Path to the top-level Facebook export directory,
                e.g. /home/user/downloads/facebook-username-2025-04-24-xxxxx/
            chunker: Optional chunker instance. Defaults to SentenceAwareChunker().
        """
        self.export_dir = Path(export_dir)
        self.messages_dir = self.export_dir / 'your_facebook_activity' / 'messages'
        self.chunker = chunker or _DEFAULT_CHUNKER
        self.query_gen = _DEFAULT_QUERY_GEN

    def source_name(self) -> str:
        return 'facebook'

    def parse(self) -> Iterator[SADocument]:
        """Walk message subdirectories and yield SADocuments for each thread."""
        if not self.messages_dir.is_dir():
            logger.warning("Facebook messages dir not found: %s", self.messages_dir)
            return

        for subdir_name in THREAD_SUBDIRS:
            subdir = self.messages_dir / subdir_name
            if not subdir.is_dir():
                continue

            for thread_dir in sorted(subdir.iterdir()):
                if not thread_dir.is_dir():
                    continue

                msg_path = thread_dir / 'message_1.json'
                if msg_path.exists():
                    try:
                        yield from self._parse_thread(msg_path)
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning(
                            "Skipping malformed thread %s: %s", msg_path, exc
                        )

    def _parse_thread(self, msg_path: Path) -> Iterator[SADocument]:
        """Parse a single thread's message_1.json into SADocuments.

        Args:
            msg_path: Path to the message_1.json file.
        """
        with open(msg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        title = self._fix_encoding(data.get('title', ''))
        participants = [
            self._fix_encoding(p['name']) for p in data.get('participants', [])
        ]

        # Extract thread ID from thread_path or directory name
        thread_path = data.get('thread_path', '')
        thread_id = thread_path.rsplit('/', 1)[-1] if thread_path else msg_path.parent.name

        for msg in data.get('messages', []):
            content = msg.get('content', '')
            if not content:
                continue

            content = self._fix_encoding(content)
            sender = self._fix_encoding(msg.get('sender_name', 'Unknown'))

            # Skip very short messages (reactions, single-word acks, etc.)
            if len(content) < 10:
                continue

            timestamp_ms = msg.get('timestamp_ms', 0)
            timestamp = timestamp_ms / 1000.0

            # Don't prepend sender to content - it causes sender names
            # to dominate entity extraction (e.g. "Gary Duncan" 50K+ times)
            chunks = self.chunker.chunk(content)

            for chunk_idx, chunk in enumerate(chunks):
                query = self.query_gen.for_facebook(sender, chunk)

                doc_id = f"fb-{thread_id[:16]}-{timestamp_ms}-{chunk_idx}"

                yield SADocument(
                    content=chunk,
                    doc_id=doc_id,
                    value=chunk[:200],
                    query=query,
                    source='facebook',
                    timestamp=timestamp,
                    metadata={
                        'thread_title': title,
                        'sender': sender,
                        'participants': participants,
                    },
                )

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Fix Facebook's Latin-1 mojibake encoding.

        Facebook exports store UTF-8 byte sequences as Latin-1 codepoints.
        For example, an emoji like U+1F60A gets stored as its UTF-8 bytes
        (\\xf0\\x9f\\x98\\x8a) each interpreted as Latin-1 characters.

        This reverses the process: encode back to Latin-1 bytes, then decode
        as UTF-8 to recover the original text.
        """
        try:
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text
