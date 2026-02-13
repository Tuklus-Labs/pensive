"""Parsers for Google Takeout data: Calendar, YouTube comments, Maps.

Reads data from a standard Google Takeout export directory structure:
  - Calendar/*.ics           (ICS calendar files)
  - YouTube and YouTube Music/comments/*.csv  (comment CSVs)
  - Maps/My labeled places/Labeled places.json (GeoJSON)
"""
import csv
import json
import logging
import re
from pathlib import Path
from typing import Iterator

from ..base import SADocument, BaseParser
from ..query_gen import QueryGenerator

logger = logging.getLogger(__name__)


class GoogleCalendarParser(BaseParser):
    """Parse Google Calendar ICS exports into SADocuments.

    Manually parses ICS (iCalendar) format without external libraries.
    Extracts VEVENT blocks and produces one SADocument per event.
    """

    def __init__(self, takeout_dir: str):
        self.takeout_dir = Path(takeout_dir)
        self.calendar_dir = self.takeout_dir / 'Calendar'
        self.query_gen = QueryGenerator()

    def source_name(self) -> str:
        return 'google_calendar'

    def parse(self) -> Iterator[SADocument]:
        """Yield SADocuments from all .ics files in the Calendar directory."""
        if not self.calendar_dir.is_dir():
            logger.warning("Calendar directory not found: %s", self.calendar_dir)
            return

        ics_files = list(self.calendar_dir.glob('*.ics'))
        if not ics_files:
            logger.warning("No .ics files found in %s", self.calendar_dir)
            return

        logger.info("Found %d .ics files in %s", len(ics_files), self.calendar_dir)

        for ics_path in ics_files:
            yield from self._parse_ics(ics_path)

    def _parse_ics(self, path: Path) -> Iterator[SADocument]:
        """Parse a single ICS file and yield SADocuments for each VEVENT."""
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError as e:
            logger.error("Failed to read %s: %s", path, e)
            return

        # Split on VEVENT boundaries
        events = text.split('BEGIN:VEVENT')
        # First element is the VCALENDAR header, skip it
        for event_block in events[1:]:
            # Trim at END:VEVENT
            end_idx = event_block.find('END:VEVENT')
            if end_idx != -1:
                event_block = event_block[:end_idx]

            doc = self._parse_vevent(event_block)
            if doc is not None:
                yield doc

    def _parse_vevent(self, block: str) -> SADocument | None:
        """Extract fields from a VEVENT block and build an SADocument."""
        fields = self._extract_ics_fields(block)

        summary = fields.get('SUMMARY', '').strip()
        if not summary:
            return None

        uid = fields.get('UID', '').strip()
        dtstart_raw = fields.get('DTSTART', '').strip()
        dtend_raw = fields.get('DTEND', '').strip()
        location = fields.get('LOCATION', '').strip()
        description = fields.get('DESCRIPTION', '').strip()

        date = self._parse_ics_date(dtstart_raw)
        if not date:
            date = 'unknown date'

        # Build content string
        content_parts = [f"Calendar event: {summary} on {date}"]
        if location:
            content_parts.append(f"Location: {location}")
        if description:
            # ICS descriptions use \\n for newlines
            clean_desc = description.replace('\\n', ' ').replace('\\,', ',')
            content_parts.append(f"Description: {clean_desc}")

        content = '. '.join(content_parts)

        # Build stable doc_id from UID
        if uid:
            doc_id = f"gcal-{uid[:16]}"
        else:
            # Fallback: hash summary + date
            import hashlib
            raw = f"{summary}:{dtstart_raw}"
            doc_id = f"gcal-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

        return SADocument(
            content=content,
            doc_id=doc_id,
            value=f"{summary} ({date})",
            query=self.query_gen.for_calendar(summary, date),
            source='google_calendar',
            metadata={
                'summary': summary,
                'date': date,
                'location': location,
            },
        )

    @staticmethod
    def _extract_ics_fields(block: str) -> dict:
        """Extract key-value pairs from an ICS block.

        Handles ICS line unfolding (continuation lines start with a space/tab)
        and property parameters (e.g. DTSTART;VALUE=DATE:20240115).
        """
        # Unfold continuation lines (RFC 5545 section 3.1)
        unfolded = re.sub(r'\r?\n[ \t]', '', block)

        fields = {}
        for line in unfolded.splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue

            # Split on first colon - property name may include parameters
            prop_part, _, value = line.partition(':')

            # Strip parameters (e.g. DTSTART;VALUE=DATE -> DTSTART)
            prop_name = prop_part.split(';')[0].upper()

            fields[prop_name] = value

        return fields

    @staticmethod
    def _parse_ics_date(raw: str) -> str | None:
        """Parse ICS date formats into human-readable strings.

        Handles:
          - YYYYMMDD (date only)
          - YYYYMMDDTHHMMSS (local datetime)
          - YYYYMMDDTHHMMSSZ (UTC datetime)
        """
        if not raw:
            return None

        # Strip trailing Z and any TZID prefix value
        clean = raw.rstrip('Z').strip()

        # Date only: 20240115
        if re.match(r'^\d{8}$', clean):
            return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"

        # DateTime: 20240115T093000
        m = re.match(r'^(\d{8})T(\d{6})$', clean)
        if m:
            d, t = m.groups()
            return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}"

        # Fallback: return as-is
        return clean if clean else None


class YouTubeCommentsParser(BaseParser):
    """Parse YouTube comments from Google Takeout CSV exports.

    Looks for CSV files in ``YouTube and YouTube Music/comments/``.
    Handles various column name conventions found in Takeout exports.
    """

    # Possible column names for the comment text
    _COMMENT_COLUMNS = {
        'Comment Text', 'comment_text', 'Comment', 'comment',
        'CommentText', 'Content', 'content', 'Text', 'text',
    }

    # Possible column names for the timestamp
    _TIMESTAMP_COLUMNS = {
        'Comment Create Timestamp', 'Timestamp', 'timestamp',
        'Created At', 'created_at', 'Date', 'date', 'Time', 'time',
    }

    def __init__(self, takeout_dir: str):
        self.takeout_dir = Path(takeout_dir)
        self.comments_dir = self.takeout_dir / 'YouTube and YouTube Music' / 'comments'
        self.query_gen = QueryGenerator()

    def source_name(self) -> str:
        return 'youtube_comments'

    def parse(self) -> Iterator[SADocument]:
        """Yield SADocuments from all CSV files in the comments directory."""
        if not self.comments_dir.is_dir():
            logger.warning("YouTube comments directory not found: %s", self.comments_dir)
            return

        csv_files = list(self.comments_dir.glob('*.csv'))
        if not csv_files:
            logger.warning("No CSV files found in %s", self.comments_dir)
            return

        logger.info("Found %d comment CSV files in %s", len(csv_files), self.comments_dir)

        for csv_path in csv_files:
            yield from self._parse_csv(csv_path)

    def _parse_csv(self, path: Path) -> Iterator[SADocument]:
        """Parse a single CSV file of YouTube comments."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    logger.warning("Empty CSV file: %s", path)
                    return

                # Find the comment text column
                comment_col = self._find_column(reader.fieldnames, self._COMMENT_COLUMNS)
                timestamp_col = self._find_column(reader.fieldnames, self._TIMESTAMP_COLUMNS)

                if comment_col is None:
                    logger.warning(
                        "No comment text column found in %s (columns: %s)",
                        path, reader.fieldnames,
                    )
                    return

                for row_idx, row in enumerate(reader):
                    comment = (row.get(comment_col) or '').strip()
                    if not comment:
                        continue

                    timestamp = (row.get(timestamp_col) or '') if timestamp_col else ''

                    # Extract key terms for the query
                    key_terms = self.query_gen._extract_key_terms(comment)
                    terms_str = ' '.join(key_terms)
                    query = f"What did I comment on YouTube? {terms_str}"

                    import hashlib
                    raw = f"{comment}:{timestamp}:{row_idx}"
                    doc_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]
                    doc_id = f"ytcomment-{doc_hash}"

                    yield SADocument(
                        content=comment,
                        doc_id=doc_id,
                        value=comment[:200],
                        query=query,
                        source='youtube_comments',
                        metadata={
                            'timestamp': timestamp,
                            'csv_file': path.name,
                        },
                    )
        except OSError as e:
            logger.error("Failed to read %s: %s", path, e)

    @staticmethod
    def _find_column(fieldnames: list, candidates: set) -> str | None:
        """Find the first matching column name from a set of candidates."""
        for name in fieldnames:
            if name in candidates:
                return name
        return None


class GoogleMapsParser(BaseParser):
    """Parse Google Maps labeled places from Takeout GeoJSON export.

    Reads ``Maps/My labeled places/Labeled places.json``.
    """

    def __init__(self, takeout_dir: str):
        self.takeout_dir = Path(takeout_dir)
        self.places_path = (
            self.takeout_dir / 'Maps' / 'My labeled places' / 'Labeled places.json'
        )

    def source_name(self) -> str:
        return 'google_maps'

    def parse(self) -> Iterator[SADocument]:
        """Yield SADocuments from each feature in the GeoJSON file."""
        if not self.places_path.is_file():
            logger.warning("Labeled places file not found: %s", self.places_path)
            return

        try:
            with open(self.places_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load %s: %s", self.places_path, e)
            return

        features = data.get('features', [])
        if not features:
            logger.warning("No features found in %s", self.places_path)
            return

        logger.info("Found %d labeled places", len(features))

        for idx, feature in enumerate(features):
            doc = self._parse_feature(feature, idx)
            if doc is not None:
                yield doc

    @staticmethod
    def _parse_feature(feature: dict, idx: int) -> SADocument | None:
        """Convert a GeoJSON feature into an SADocument."""
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        name = (properties.get('name') or properties.get('title') or '').strip()
        address = (properties.get('address') or properties.get('location') or '').strip()

        if not name and not address:
            return None

        # Build content
        display_name = name or 'Unnamed place'
        if address:
            content = f"Saved place: {display_name} at {address}"
        else:
            content = f"Saved place: {display_name}"

        # Extract coordinates if available
        coords = geometry.get('coordinates', [])
        metadata = {'address': address}
        if coords and len(coords) >= 2:
            metadata['longitude'] = coords[0]
            metadata['latitude'] = coords[1]

        import hashlib
        raw = f"{name}:{address}:{idx}"
        doc_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]

        return SADocument(
            content=content,
            doc_id=f"gmaps-{doc_hash}",
            value=display_name,
            query=f"Where is {display_name}?",
            source='google_maps',
            metadata=metadata,
        )
