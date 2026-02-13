"""Parser for ChatGPT export data (conversations.json).

Reads the full conversations.json from a ChatGPT data export, extracts
the active thread from each conversation (latest leaf → root), chunks
message text, and yields SADocuments for spreading activation ingestion.

Data format:
  - Top-level: JSON array of conversation objects
  - Each conversation: title, create_time, update_time, mapping
  - mapping: dict of UUID → node {id, message, parent, children}
  - message: {author.role, content.content_type, content.parts, create_time, metadata.model_slug}
"""
import hashlib
import json
import logging
import os
from typing import Dict, Iterator, List, Optional

from ..base import SADocument, BaseParser
from ..chunker import SentenceAwareChunker
from ..query_gen import QueryGenerator

logger = logging.getLogger(__name__)

# Content types that represent internal model reasoning, not user-facing text.
_SKIP_CONTENT_TYPES = frozenset({'thoughts', 'reasoning_recap'})


class ChatGPTParser(BaseParser):
    """Parse ChatGPT data export into SADocuments.

    Expects the export directory to contain ``conversations.json``.
    For each conversation, follows the latest thread (latest leaf node
    back to root) and yields chunked documents for every message with
    meaningful text content.
    """

    def __init__(
        self,
        export_dir: str,
        chunker: SentenceAwareChunker = None,
    ):
        self.export_dir = export_dir
        self.chunker = chunker or SentenceAwareChunker()
        self.query_gen = QueryGenerator()

    def source_name(self) -> str:
        return 'chatgpt'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Iterator[SADocument]:
        """Yield SADocuments from every conversation in the export."""
        conversations_path = os.path.join(self.export_dir, 'conversations.json')
        if not os.path.isfile(conversations_path):
            logger.error("conversations.json not found in %s", self.export_dir)
            return

        logger.info("Loading %s ...", conversations_path)
        with open(conversations_path, 'r', encoding='utf-8') as fh:
            conversations: list = json.load(fh)
        logger.info("Loaded %d conversations", len(conversations))

        for conv in conversations:
            yield from self._process_conversation(conv)

    # ------------------------------------------------------------------
    # Conversation processing
    # ------------------------------------------------------------------

    def _process_conversation(self, conv: dict) -> Iterator[SADocument]:
        """Extract the active thread and yield documents for each message."""
        title = conv.get('title') or 'Untitled'
        conv_id = self._get_conv_id(conv)
        conv_create_time = conv.get('create_time') or 0.0

        thread = self._extract_thread(conv)
        if not thread:
            return

        for msg_index, msg in enumerate(thread):
            text = msg['text']
            if len(text) <= 20:
                continue

            role = msg['role']
            timestamp = msg['timestamp'] or conv_create_time
            model_slug = msg['model_slug']

            chunks = self.chunker.chunk(text)

            for chunk_index, chunk in enumerate(chunks):
                query = self.query_gen.for_chatgpt(title, role, chunk)
                doc_id = f"chatgpt-{conv_id[:12]}-{msg_index}-{chunk_index}"
                value = chunk[:200]

                yield SADocument(
                    content=chunk,
                    doc_id=doc_id,
                    value=value,
                    query=query,
                    source='chatgpt',
                    timestamp=timestamp,
                    metadata={
                        'conv_title': title,
                        'conv_id': conv_id,
                        'role': role,
                        'model': model_slug,
                        'msg_index': msg_index,
                    },
                )

    # ------------------------------------------------------------------
    # Thread extraction
    # ------------------------------------------------------------------

    def _extract_thread(self, conv: dict) -> List[dict]:
        """Walk the conversation tree to recover the active thread.

        Strategy:
          1. Find all leaf nodes (children == []).
          2. Pick the leaf whose message has the latest create_time
             (this is the "current" conversation path after edits/branches).
          3. Walk parent links back to the root.
          4. Reverse to chronological order.
          5. Return list of message dicts with role/text/timestamp/model_slug.
        """
        mapping: Dict[str, dict] = conv.get('mapping') or {}
        if not mapping:
            return []

        # --- find leaves ---
        leaves: List[str] = []
        for node_id, node in mapping.items():
            children = node.get('children')
            if not children:  # empty list or None
                leaves.append(node_id)

        if not leaves:
            return []

        # --- pick the leaf with the latest message timestamp ---
        def _leaf_time(node_id: str) -> float:
            node = mapping.get(node_id, {})
            msg = node.get('message')
            if msg and msg.get('create_time'):
                return msg['create_time']
            return 0.0

        best_leaf = max(leaves, key=_leaf_time)

        # --- walk backward to root ---
        path_ids: List[str] = []
        current = best_leaf
        visited = set()  # guard against cycles
        while current and current not in visited:
            visited.add(current)
            path_ids.append(current)
            node = mapping.get(current, {})
            current = node.get('parent')

        path_ids.reverse()  # root → leaf order

        # --- extract message data ---
        thread: List[dict] = []
        for node_id in path_ids:
            node = mapping.get(node_id, {})
            msg = node.get('message')
            if msg is None:
                continue

            text = self._extract_text(msg)
            if text is None:
                continue

            author = msg.get('author') or {}
            role = author.get('role', 'unknown')
            # Skip system messages — they are preamble/instructions, not conversation
            if role == 'system':
                continue

            create_time = msg.get('create_time') or 0.0
            metadata = msg.get('metadata') or {}
            model_slug = metadata.get('model_slug', '')

            thread.append({
                'role': role,
                'text': text,
                'timestamp': create_time,
                'model_slug': model_slug,
            })

        return thread

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(msg: dict) -> Optional[str]:
        """Extract displayable text from a message object.

        Returns None if:
          - content is missing
          - content_type is internal reasoning (thoughts, reasoning_recap)
          - no string parts are found
        """
        content = msg.get('content')
        if not content:
            return None

        content_type = content.get('content_type', '')
        if content_type in _SKIP_CONTENT_TYPES:
            return None

        parts = content.get('parts')
        if not parts:
            return None

        # Filter to string parts only — dicts are images, tool calls, etc.
        text_parts = [p for p in parts if isinstance(p, str)]
        if not text_parts:
            return None

        text = ' '.join(text_parts).strip()
        return text if text else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_conv_id(conv: dict) -> str:
        """Extract or derive a stable conversation ID.

        Checks for ``conversation_id`` or ``id`` at the top level.
        Falls back to a hash of the title + create_time.
        """
        cid = conv.get('conversation_id') or conv.get('id')
        if cid:
            return str(cid)

        # Deterministic fallback
        title = conv.get('title') or ''
        create_time = conv.get('create_time') or 0.0
        raw = f"{title}:{create_time}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()
