"""Forward-only schema migrations for the canonical store.

This module owns the schema-version ladder. `CURRENT_SCHEMA_VERSION` is the
highest version this build knows how to produce; `store.py` imports it from
here so there is a single source of truth. Migrations never run backwards: a
database recorded at a version newer than this build is refused rather than
downgraded, because canonical data must never become unrecoverable.
"""

CURRENT_SCHEMA_VERSION = 4

_V4_DDL = (
    """CREATE TABLE IF NOT EXISTS task_checkpoints (
  id             TEXT PRIMARY KEY,
  project        TEXT NOT NULL,
  agent          TEXT NOT NULL,
  task_id        TEXT NOT NULL,
  revision       INTEGER NOT NULL CHECK(revision >= 1),
  request_id     TEXT NOT NULL UNIQUE,
  state          TEXT NOT NULL CHECK(state IN ('active','blocked','completed','abandoned')),
  body           TEXT NOT NULL,
  source         TEXT NOT NULL,
  writer_session TEXT,
  source_ref     TEXT,
  recorded_at    INTEGER NOT NULL,
  UNIQUE(project, agent, task_id, revision)
)""",
    """CREATE INDEX IF NOT EXISTS idx_task_checkpoints_scope_revision
  ON task_checkpoints(project, agent, task_id, revision DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_task_checkpoints_task_agent_project
  ON task_checkpoints(task_id, agent, project)""",
    """CREATE TABLE IF NOT EXISTS recall_receipts (
  id          TEXT PRIMARY KEY,
  query       TEXT NOT NULL,
  project     TEXT,
  agent       TEXT NOT NULL,
  task_id     TEXT NOT NULL,
  source_ref  TEXT NOT NULL,
  recorded_at INTEGER NOT NULL
)""",
    """CREATE INDEX IF NOT EXISTS idx_recall_receipts_task_agent_time
  ON recall_receipts(task_id, agent, recorded_at)""",
    """CREATE TABLE IF NOT EXISTS recall_exposures (
  receipt_id TEXT NOT NULL REFERENCES recall_receipts(id),
  atom_id    TEXT NOT NULL REFERENCES atoms(id),
  rank       INTEGER NOT NULL CHECK(rank >= 1),
  score      REAL,
  delivery   TEXT NOT NULL,
  PRIMARY KEY(receipt_id, atom_id),
  UNIQUE(receipt_id, rank)
)""",
    """CREATE INDEX IF NOT EXISTS idx_recall_exposures_atom
  ON recall_exposures(atom_id)""",
    """CREATE TABLE IF NOT EXISTS recall_feedback (
  event_id      TEXT PRIMARY KEY,
  receipt_id    TEXT NOT NULL,
  atom_id       TEXT NOT NULL,
  feedback_type TEXT NOT NULL CHECK(feedback_type IN ('shown','used','helpful','irrelevant','outdated')),
  source        TEXT NOT NULL,
  agent         TEXT NOT NULL,
  task_id       TEXT NOT NULL,
  session_id    TEXT,
  source_ref    TEXT,
  note          TEXT,
  recorded_at   INTEGER NOT NULL,
  processed_at  INTEGER,
  FOREIGN KEY(receipt_id, atom_id)
    REFERENCES recall_exposures(receipt_id, atom_id)
)""",
    """CREATE INDEX IF NOT EXISTS idx_recall_feedback_pending_type
  ON recall_feedback(processed_at, feedback_type)""",
    """CREATE INDEX IF NOT EXISTS idx_recall_feedback_receipt
  ON recall_feedback(receipt_id)""",
    """CREATE TABLE IF NOT EXISTS memory_credits (
  atom_id     TEXT NOT NULL REFERENCES atoms(id),
  agent       TEXT NOT NULL,
  task_id     TEXT NOT NULL,
  feedback_id TEXT NOT NULL REFERENCES recall_feedback(event_id),
  awarded_at  INTEGER NOT NULL,
  PRIMARY KEY(atom_id, agent, task_id)
)""",
)


def migrate(store):
    """Advance ``store`` from its recorded schema version up to current.

    A fresh database (no recorded version) is treated as version 0. The v0->v1
    step is a no-op beyond stamping the version, because ``schema.sql`` already
    installed the v1 tables. Later versions add their upgrade step here.
    """
    version = store._readVersion()
    if version is None:
        version = 0

    if version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema_version {version} is newer than this build "
            f"supports (max {CURRENT_SCHEMA_VERSION}); refusing to open"
        )

    dirty = False
    if version < 1:
        # v0 -> v1: baseline. Tables already exist; record that we are at v1.
        store._setVersion(1)
        version = 1
        dirty = True

    if version < 2:
        _upgrade_1_to_2(store)
        store._setVersion(2)
        version = 2
        dirty = True

    if version < 3:
        _upgrade_2_to_3(store)
        store._setVersion(3)
        version = 3
        dirty = True

    if version < 4:
        # sqlite3.executescript commits an open transaction before running. Use
        # individual DDL statements for this step so the schema objects and the
        # version stamp are one atomic upgrade.
        try:
            # The older ladder stamps each version with a normal INSERT. On a
            # fresh file, that leaves the v3 stamp's transaction open after the
            # v2/v3 executescript calls; close that completed prefix before
            # starting the v4 write lock.
            if store._conn.in_transaction:
                store._commit()
            store._conn.execute("BEGIN IMMEDIATE")
            _upgrade_3_to_4(store)
            store._setVersion(4)
            version = 4
            dirty = True
        except Exception:
            store._conn.rollback()
            raise

    if dirty:  # a no-op re-open of an up-to-date db writes nothing
        store._commit()


def _upgrade_1_to_2(store):
    """Add lifecycle job tables and the provenance source_ref hot-path index."""
    store._conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_prov_source_ref ON provenance(source_ref);

        CREATE TABLE IF NOT EXISTS recall_log (
          id           TEXT PRIMARY KEY,
          atom_id      TEXT NOT NULL REFERENCES atoms(id),
          query        TEXT,
          source_ref   TEXT,
          weight       REAL NOT NULL DEFAULT 1.0,
          recorded_at  INTEGER NOT NULL,
          processed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_recall_log_unprocessed
          ON recall_log(processed_at, atom_id);

        CREATE TABLE IF NOT EXISTS supersession_proposals (
          id          TEXT PRIMARY KEY,
          old_atom_id TEXT NOT NULL REFERENCES atoms(id),
          new_atom_id TEXT NOT NULL REFERENCES atoms(id),
          similarity  REAL NOT NULL,
          reason      TEXT NOT NULL,
          status      TEXT NOT NULL DEFAULT 'proposed',
          created_at  INTEGER NOT NULL,
          UNIQUE(old_atom_id, new_atom_id)
        );
        CREATE INDEX IF NOT EXISTS idx_supersession_proposals_status
          ON supersession_proposals(status);
        """
    )


def _upgrade_2_to_3(store):
    """Add the covering facet index the L1 lookup route needs to avoid sorting.

    ``serve.l1.handleLookup`` filters facets on (key, value) and returns
    atom_ids ORDER BY atom_id. ``idx_facets_kv`` is on (key, value) alone and
    cannot produce that order, so SQLite materialized every matching row into a
    temp B-tree and sorted it BEFORE applying the LIMIT -- the LIMIT bounded the
    response, not the work. Since /lookup is unauthenticated and runs inline on
    the daemon's event-loop thread, and facet values are guessable rather than
    secret, one request against a hot value stalled every other caller.
    Measured on a 1.25M-row fixture with 200k rows under one value: 58ms per
    lookup before, 0.08ms after.

    Purely additive, which is what makes it safe to run against the existing
    1.6GB store: it creates an index and touches no row. Measured on a
    1.25M-row facets table it takes ~0.4s and adds ~63MB to the database
    (~64MB through the WAL); the store is unreadable for that moment and
    correct either side of it, and a re-open once the index exists is a no-op
    at ~0.05ms.

    ``idx_facets_kv`` is now a strict prefix of this index and therefore
    redundant, but it is deliberately LEFT IN PLACE: dropping it is a separate,
    non-additive decision about write cost, and this migration's job is to make
    reads safe without putting an existing store at risk.
    """
    store._conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_facets_kv_atom
          ON facets(key, value, atom_id);
        """
    )


def _upgrade_3_to_4(store):
    """Create task state and feedback tables without changing old rows."""
    for ddl in _V4_DDL:
        store._conn.execute(ddl)
