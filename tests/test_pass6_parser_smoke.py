"""Smoke tests for ingestion parsers and the SentenceAwareChunker.

PENPY-P6-MIN-2: pass-6 audit found that FacebookParser,
GoogleCalendarParser, YouTubeCommentsParser, GoogleMapsParser, and
SentenceAwareChunker had zero direct unit tests. ChatGPT and Email
parsers were covered by pass-1; the rest were exercised only
indirectly (or not at all).

These are coverage smoke tests: one test per class, happy path,
synthetic minimal input. They verify the parser doesn't crash and
returns documents with the expected shape (doc_id, content, source).
Exhaustive correctness (encoding edge cases, partial files, etc.) is
deliberately out of scope -- the goal is to catch contract regressions
quickly without claiming full coverage.
"""
import csv
import json
from pathlib import Path

import pytest

from pensive.ingestion.base import SADocument
from pensive.ingestion.chunker import SentenceAwareChunker
from pensive.ingestion.parsers.facebook import FacebookParser
from pensive.ingestion.parsers.google import (
    GoogleCalendarParser,
    GoogleMapsParser,
    YouTubeCommentsParser,
)


# FacebookParser


def test_facebook_parser_yields_documents_from_inbox_thread(tmp_path):
    """A minimal Facebook export with one inbox thread must produce at
    least one SADocument tagged source='facebook' with non-empty content."""
    inbox_dir = (
        tmp_path / 'your_facebook_activity' / 'messages' / 'inbox' / 'thread_alice'
    )
    inbox_dir.mkdir(parents=True)
    msg_data = {
        'title': 'Alice',
        'participants': [{'name': 'Gary'}, {'name': 'Alice'}],
        'thread_path': 'inbox/thread_alice_123',
        'messages': [
            {
                'sender_name': 'Alice',
                'timestamp_ms': 1_700_000_000_000,
                'content': 'Hello Gary, the meeting is rescheduled to Tuesday.',
            },
            {
                'sender_name': 'Gary',
                'timestamp_ms': 1_700_000_060_000,
                'content': 'Acknowledged, Tuesday it is.',
            },
        ],
    }
    (inbox_dir / 'message_1.json').write_text(json.dumps(msg_data))

    parser = FacebookParser(str(tmp_path))
    docs = list(parser.parse())

    assert docs, "FacebookParser yielded zero documents from a valid thread"
    for d in docs:
        assert isinstance(d, SADocument), f"expected SADocument, got {type(d)}"
        assert d.source == 'facebook', f"expected source='facebook', got {d.source!r}"
        assert d.doc_id.startswith('fb-'), f"unexpected doc_id format: {d.doc_id!r}"
        assert d.content, "FacebookParser yielded empty content"


# GoogleCalendarParser


def test_google_calendar_parser_yields_event_documents(tmp_path):
    """A minimal Calendar ICS file with one VEVENT must produce one
    SADocument with source='google_calendar'."""
    cal_dir = tmp_path / 'Calendar'
    cal_dir.mkdir()
    ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:event-001@example.com\r\n"
        "DTSTART:20260115T093000\r\n"
        "DTEND:20260115T103000\r\n"
        "SUMMARY:Project standup\r\n"
        "LOCATION:Conference Room A\r\n"
        "DESCRIPTION:Weekly sync on roadmap.\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    (cal_dir / 'gary.ics').write_text(ics)

    parser = GoogleCalendarParser(str(tmp_path))
    docs = list(parser.parse())

    assert len(docs) == 1, f"expected 1 event document, got {len(docs)}"
    d = docs[0]
    assert d.source == 'google_calendar', f"got source={d.source!r}"
    assert d.doc_id.startswith('gcal-'), f"unexpected doc_id: {d.doc_id!r}"
    assert 'Project standup' in d.content, (
        f"event summary not in content: {d.content!r}"
    )


# YouTubeCommentsParser


def test_youtube_comments_parser_yields_documents_from_csv(tmp_path):
    """A minimal YouTube-comments CSV must produce one SADocument per
    non-empty row with source='youtube_comments'."""
    comments_dir = tmp_path / 'YouTube and YouTube Music' / 'comments'
    comments_dir.mkdir(parents=True)
    csv_path = comments_dir / 'comments.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(
            fh, fieldnames=['Comment Text', 'Comment Create Timestamp']
        )
        writer.writeheader()
        writer.writerow({
            'Comment Text': 'Great video, the explanation of FAISS was clear.',
            'Comment Create Timestamp': '2026-01-15T10:00:00Z',
        })
        writer.writerow({
            'Comment Text': 'Could you cover IVF tuning next?',
            'Comment Create Timestamp': '2026-01-15T11:00:00Z',
        })

    parser = YouTubeCommentsParser(str(tmp_path))
    docs = list(parser.parse())

    assert len(docs) == 2, f"expected 2 comment documents, got {len(docs)}"
    for d in docs:
        assert d.source == 'youtube_comments', f"got source={d.source!r}"
        assert d.doc_id.startswith('ytcomment-'), (
            f"unexpected doc_id: {d.doc_id!r}"
        )
        assert d.content, "YouTubeCommentsParser yielded empty content"


# GoogleMapsParser


def test_google_maps_parser_yields_documents_from_geojson(tmp_path):
    """A minimal labeled-places GeoJSON must produce one SADocument per
    feature with source='google_maps'."""
    maps_dir = tmp_path / 'Maps' / 'My labeled places'
    maps_dir.mkdir(parents=True)
    geojson = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {
                    'name': 'Home',
                    'address': '123 Farm Road, Centralia, WA',
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [-122.95, 46.71],
                },
            },
            {
                'type': 'Feature',
                'properties': {
                    'name': 'Workshop',
                    'address': '456 Forge Lane, Centralia, WA',
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [-122.96, 46.72],
                },
            },
        ],
    }
    (maps_dir / 'Labeled places.json').write_text(json.dumps(geojson))

    parser = GoogleMapsParser(str(tmp_path))
    docs = list(parser.parse())

    assert len(docs) == 2, f"expected 2 place documents, got {len(docs)}"
    for d in docs:
        assert d.source == 'google_maps', f"got source={d.source!r}"
        assert d.doc_id.startswith('gmaps-'), (
            f"unexpected doc_id: {d.doc_id!r}"
        )
        assert d.content.startswith('Saved place:'), (
            f"unexpected content prefix: {d.content!r}"
        )


# SentenceAwareChunker


def test_sentence_aware_chunker_returns_all_sentences():
    """A 3-sentence input over target_size must split into multiple chunks
    that collectively cover all three sentences."""
    s1 = "The first sentence describes the morning routine in detail."
    s2 = "The second sentence covers the long meeting that took place at noon."
    s3 = "The third sentence wraps up the day with an evening reflection."
    text = f"{s1} {s2} {s3}"

    # target_size smaller than the full text but larger than any one sentence
    # forces splitting without dropping sentences.
    chunker = SentenceAwareChunker(target_size=120, overlap=20, min_size=10)
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2, (
        f"expected multi-chunk split for {len(text)}-char input, got {len(chunks)}"
    )
    joined = ' '.join(chunks)
    # All three sentences must appear somewhere across the chunks.
    for sent in (s1, s2, s3):
        # Each sentence is one logical unit; strip trailing period for the
        # substring match (the chunker uses re.split which discards
        # whitespace but keeps the punctuation on the preceding sentence).
        head = sent.split('.')[0]
        assert head in joined, (
            f"chunker dropped sentence prefix {head!r}: chunks={chunks!r}"
        )


def test_sentence_aware_chunker_short_input_single_chunk():
    """An input below target_size must return a single chunk unchanged."""
    chunker = SentenceAwareChunker(target_size=800, overlap=100, min_size=50)
    text = "Short input under the threshold."
    chunks = chunker.chunk(text)
    assert chunks == [text], (
        f"expected single-chunk passthrough for short input, got {chunks!r}"
    )
